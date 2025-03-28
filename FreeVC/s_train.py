import logging
import os

import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from commons import clip_grad_value_, slice_segments
from constants import (
    DATA_FILTER_LENGTH,
    DATA_HOP_LENGTH,
    DATA_MAX_WAV_VALUE,
    DATA_MEL_FMAX,
    DATA_MEL_FMIN,
    DATA_N_MEL_CHANNELS,
    DATA_SAMPLING_RATE,
    DATA_TRAINING_FILES,
    DATA_VALIDATION_FILES,
    DATA_WIN_LENGTH,
    MODEL_GIN_CHANNELS,
    MODEL_HIDDEN_CHANNELS,
    MODEL_INTER_CHANNELS,
    MODEL_RESBLOCK,
    MODEL_RESBLOCK_DILATION_SIZES,
    MODEL_RESBLOCK_KERNEL_SIZES,
    MODEL_SSL_DIM,
    MODEL_UPSAMPLE_INITIAL_CHANNEL,
    MODEL_UPSAMPLE_KERNEL_SIZES,
    MODEL_UPSAMPLE_RATES,
    MODEL_USE_SPECTRAL_NORM,
    MODEL_USE_SPK,
    TRAIN_BATCH_SIZE,
    TRAIN_BETAS,
    TRAIN_C_KL,
    TRAIN_C_MEL,
    TRAIN_EPOCHS,
    TRAIN_EPS,
    TRAIN_FP16_RUN,
    TRAIN_LEARNING_RATE,
    TRAIN_LR_DECAY,
    TRAIN_MAX_SPECLEN,
    TRAIN_SEGMENT_SIZE,
    TRAIN_USE_SR,
)
from dataloaders import TextAudioSpeakerCollate, TextAudioSpeakerLoader
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import MultiPeriodDiscriminator, SynthesizerTrn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)
best_loss = float("inf")

wandb.init(project="VoiceConversion")
global_step = 0


def main():
    global best_loss
    assert torch.cuda.is_available(), "CPU training is not allowed."
    device = torch.device("cuda:0")
    torch.manual_seed(1234)

    train_dataset = TextAudioSpeakerLoader(
        DATA_TRAINING_FILES,
        max_wav_value=DATA_MAX_WAV_VALUE,
        sampling_rate=DATA_SAMPLING_RATE,
        filter_length=DATA_FILTER_LENGTH,
        hop_length=DATA_HOP_LENGTH,
        win_length=DATA_WIN_LENGTH,
        use_sr=TRAIN_USE_SR,
        use_spk=MODEL_USE_SPK,
        max_speclen=TRAIN_MAX_SPECLEN,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=TextAudioSpeakerCollate(
            DATA_HOP_LENGTH, MODEL_USE_SPK, DATA_HOP_LENGTH, TRAIN_MAX_SPECLEN
        ),
    )

    eval_dataset = TextAudioSpeakerLoader(
        DATA_VALIDATION_FILES,
        max_wav_value=DATA_MAX_WAV_VALUE,
        sampling_rate=DATA_SAMPLING_RATE,
        filter_length=DATA_FILTER_LENGTH,
        hop_length=DATA_HOP_LENGTH,
        win_length=DATA_WIN_LENGTH,
        use_sr=TRAIN_USE_SR,
        use_spk=MODEL_USE_SPK,
        max_speclen=TRAIN_MAX_SPECLEN,
    )
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=8,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        collate_fn=TextAudioSpeakerCollate(
            DATA_HOP_LENGTH, MODEL_USE_SPK, DATA_HOP_LENGTH, TRAIN_MAX_SPECLEN
        ),
    )

    net_g = SynthesizerTrn(
        DATA_FILTER_LENGTH // 2 + 1,
        TRAIN_SEGMENT_SIZE // DATA_HOP_LENGTH,
        inter_channels=MODEL_INTER_CHANNELS,
        hidden_channels=MODEL_HIDDEN_CHANNELS,
        resblock=MODEL_RESBLOCK,
        resblock_kernel_sizes=MODEL_RESBLOCK_KERNEL_SIZES,
        resblock_dilation_sizes=MODEL_RESBLOCK_DILATION_SIZES,
        upsample_rates=MODEL_UPSAMPLE_RATES,
        upsample_initial_channel=MODEL_UPSAMPLE_INITIAL_CHANNEL,
        upsample_kernel_sizes=MODEL_UPSAMPLE_KERNEL_SIZES,
        gin_channels=MODEL_GIN_CHANNELS,
        ssl_dim=MODEL_SSL_DIM,
    ).to(device)
    net_d = MultiPeriodDiscriminator(MODEL_USE_SPECTRAL_NORM).to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(), TRAIN_LEARNING_RATE, betas=TRAIN_BETAS, eps=TRAIN_EPS
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), TRAIN_LEARNING_RATE, betas=TRAIN_BETAS, eps=TRAIN_EPS
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=TRAIN_LR_DECAY)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=TRAIN_LR_DECAY)

    scaler = GradScaler(enabled=TRAIN_FP16_RUN)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        loss = train(
            device,
            net_g,
            net_d,
            optim_g,
            optim_d,
            scaler,
            train_loader,
            epoch,
        )
        # eval_loss = evaluate(net_g, eval_loader, device)
        # wandb.log({"epoch": epoch, "eval_loss": eval_loss})
        if loss < best_loss:
            save_checkpoint(epoch, net_g, optim_g, loss)
            best_loss = loss
        scheduler_g.step()
        scheduler_d.step()


def train(device, net_g, net_d, optim_g, optim_d, scaler, train_loader, epoch):
    global global_step

    net_g.train()
    net_d.train()

    for batch_idx, (c, spec, y, filenames) in enumerate(train_loader):
        spec, y, c = spec.to(device), y.to(device), c.to(device)

        mel = spec_to_mel_torch(
            spec,
            DATA_FILTER_LENGTH,
            DATA_N_MEL_CHANNELS,
            DATA_SAMPLING_RATE,
            DATA_MEL_FMIN,
            DATA_MEL_FMAX,
        )

        # ----------------------
        # (1) Train Discriminator
        # ----------------------
        with autocast(enabled=TRAIN_FP16_RUN):
            # Generator forward pass
            y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                c, spec, mel, filenames
            )
            y_mel = slice_segments(
                mel, ids_slice, TRAIN_SEGMENT_SIZE // DATA_HOP_LENGTH
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                DATA_FILTER_LENGTH,
                DATA_N_MEL_CHANNELS,
                DATA_SAMPLING_RATE,
                DATA_HOP_LENGTH,
                DATA_WIN_LENGTH,
                DATA_MEL_FMIN,
                DATA_MEL_FMAX,
            )

            # Slice y to match y_hat
            y = slice_segments(y, ids_slice * DATA_HOP_LENGTH, TRAIN_SEGMENT_SIZE)

            # Discriminator forward pass
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        # ----------------------
        # (2) Train Generator
        # ----------------------
        with autocast(enabled=TRAIN_FP16_RUN):
            # Run discriminator again (separate forward pass for stability)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                # Compute losses
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * TRAIN_C_MEL
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * TRAIN_C_KL
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)

                # Combine generator losses
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        scaler.step(optim_g)
        scaler.update()

        wandb.log(
            {
                "loss/discriminator": loss_disc.item(),
                "loss/generator": loss_gen.item(),
                "learning_rate": optim_g.param_groups[0]["lr"],
                "epoch": epoch,
                "step": global_step,
            }
        )
    return loss_gen_all.item()


def evaluate(generator, eval_loader, device):
    generator.eval()
    total_loss = 0.0

    with torch.no_grad():
        for c, spec, y, filenames in eval_loader:
            c, spec, y = c.to(device), spec.to(device), y.to(device)
            mel = spec_to_mel_torch(
                spec,
                DATA_FILTER_LENGTH,
                DATA_N_MEL_CHANNELS,
                DATA_SAMPLING_RATE,
                DATA_MEL_FMIN,
                DATA_MEL_FMAX,
            )

            with torch.cuda.amp.autocast(enabled=True):
                y_hat, ids_slice, _, _ = generator.infer(c, mel, filenames)
                y = slice_segments(y, ids_slice * DATA_HOP_LENGTH, TRAIN_SEGMENT_SIZE)

                loss = torch.nn.functional.l1_loss(y, y_hat)
                total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    wandb.log({"eval_loss": avg_loss})
    logging.info(f"Evaluation completed. Avg Loss: {avg_loss:.6f}")
    return avg_loss


def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    best_path = os.path.join(log_dir, "best_model.pth")
    torch.save(model.state_dict(), best_path)
    best_data = os.path.join(log_dir, "best_data.pth")
    torch.save(checkpoint, best_data)
    logging.info(f"Best checkpoint updated: {best_path}")


if __name__ == "__main__":
    main()
