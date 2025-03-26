import argparse
import logging
import os
import time

import librosa
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

from constants import (
    DATA_FILTER_LENGTH,
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
    TRAIN_SEGMENT_SIZE,
)
from models.synthesizer import SynthesizerTrn, extract_speaker_embedding
from utils import get_cmodel, get_content, get_hparams_from_file

logging.getLogger("numba").setLevel(logging.WARNING)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "module" in list(checkpoint.keys())[0]:
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpfile",
        type=str,
        default="configs/freevc.json",
        help="path to json config file",
    )
    parser.add_argument(
        "--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file"
    )
    parser.add_argument(
        "--txtpath", type=str, default="convert.txt", help="path to txt file"
    )
    parser.add_argument(
        "--outdir", type=str, default="output/freevc", help="path to output dir"
    )
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hps = get_hparams_from_file(args.hpfile)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    net_g = SynthesizerTrn(
        DATA_FILTER_LENGTH // 2 + 1,
        TRAIN_SEGMENT_SIZE // DATA_FILTER_LENGTH,
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
    ).cuda(device)
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = load_checkpoint("checkpoints/best_model.pth", net_g)

    print("Loading WavLM for content...")
    cmodel = get_cmodel(0)

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

            g_tgt = extract_speaker_embedding(tgt).to(device)
            g_tgt = g_tgt.unsqueeze(0).unsqueeze(-1)  # Match expected shape

            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).to(device).unsqueeze(0)
            c = get_content(cmodel, wav_src).to(device)

            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(device)

            audio = net_g.infer(
                c.to(device), filenames=[tgt], c_lengths=c_lengths
            )  # Pass everything to CUDA

            audio = audio[0][0].data.cpu().float().numpy()

            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(
                    os.path.join(args.outdir, f"{timestamp}_{title}.wav"),
                    hps.data.sampling_rate,
                    audio,
                )
            else:
                write(
                    os.path.join(args.outdir, f"{title}.wav"),
                    hps.data.sampling_rate,
                    audio,
                )
