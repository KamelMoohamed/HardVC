import torch
import torch.nn as nn

from commons import rand_slice_segments
from DeepSpeaker.deep_speaker.audio import read_mfcc
from DeepSpeaker.deep_speaker.batcher import sample_from_mfcc
from DeepSpeaker.deep_speaker.constants import NUM_FRAMES, SAMPLE_RATE
from DeepSpeaker.deep_speaker.conv_models import DeepSpeakerModel
from models.speaker_encoder import SpeakerEncoder

from .encoder import Encoder
from .generator import Generator
from .residual_coupling_block import ResidualCouplingBlock


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training using Deep Speaker and Speaker Encoder.
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
    ):
        super().__init__()

        self.deep_speaker_model = DeepSpeakerModel()
        self.speaker_encoder = SpeakerEncoder(
            model_hidden_size=gin_channels,
            model_embedding_size=gin_channels,
        )

        self.embedding_fusion = nn.Sequential(
            nn.Linear(2 * gin_channels, gin_channels),
            nn.ReLU(),
            nn.Linear(gin_channels, gin_channels),
        )

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels

        self.enc_p = Encoder(
            ssl_dim,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = Encoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
        )

    def forward(self, c, spec, mel, filenames=None, c_lengths=None, spec_lengths=None):
        device = c.device

        if c_lengths is None:
            c_lengths = torch.ones(c.size(0), device=device) * c.size(-1)

        if spec_lengths is None:
            spec_lengths = torch.ones(spec.size(0), device=device) * spec.size(-1)

        deep_embeddings = []
        for filename in filenames:
            mfcc_np = sample_from_mfcc(read_mfcc(filename, SAMPLE_RATE), NUM_FRAMES)
            mfcc = (
                torch.from_numpy(mfcc_np).permute(2, 1, 0).unsqueeze(0).to(device)
            )  # [1, 1, D, T]
            embedding = self.deep_speaker_model(mfcc)  # [1, gin_channels]
            deep_embeddings.append(embedding)

        g_deep = torch.cat(deep_embeddings, dim=0)  # [B, gin_channels]

        mel_input = mel.transpose(1, 2)  # [B, T, mel_dim]
        g_speaker = self.speaker_encoder(mel_input)  # [B, gin_channels]

        g_fused = torch.cat([g_deep, g_speaker], dim=1)  # [B, 2 * gin_channels]
        g_fused = self.embedding_fusion(g_fused)  # [B, gin_channels]
        g = g_fused.unsqueeze(-1)  # [B, gin_channels, 1]

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c, mel, filenames=None, c_lengths=None):
        device = c.device
        c_lengths = c_lengths or torch.full((c.size(0),), c.size(-1), device=device)

        deep_embeddings = []
        for filename in filenames:
            mfcc_np = sample_from_mfcc(read_mfcc(filename, SAMPLE_RATE), NUM_FRAMES)
            mfcc = torch.from_numpy(mfcc_np).to(device)
            embedding = self.deep_speaker_model(mfcc.unsqueeze(0))  # [1, gin_channels]
            deep_embeddings.append(embedding)

        g_deep = torch.cat(deep_embeddings, dim=0)  # [B, gin_channels]

        mel_input = mel.transpose(1, 2)  # [B, T, mel_dim]
        g_speaker = self.speaker_encoder(mel_input)  # [B, gin_channels]

        g_fused = torch.cat([g_deep, g_speaker], dim=1)  # [B, 2 * gin_channels]
        g_fused = self.embedding_fusion(g_fused)  # [B, gin_channels]
        g = g_fused.unsqueeze(-1)  # [B, gin_channels, 1]

        z_p, _, _, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g)

        return o
