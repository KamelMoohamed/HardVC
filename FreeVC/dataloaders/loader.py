import os
import random

import numpy as np
import torch

from mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(
        self,
        audiopaths,
        max_wav_value,
        sampling_rate,
        filter_length,
        hop_length,
        win_length,
        use_sr,
        use_spk,
        max_speclen,
    ):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.use_sr = use_sr
        self.use_spk = use_spk
        self.spec_len = max_speclen

        random.seed(1234)
        random.shuffle(self.audiopaths)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        for audiopath in self.audiopaths:
            lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
        self.lengths = lengths

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        if self.use_spk:
            spk_filename = filename.replace(".wav", ".npy")
            spk_filename = spk_filename.replace("DUMMY", "dataset/spk")
            spk = torch.from_numpy(np.load(spk_filename))

        if not self.use_sr:
            c_filename = filename.replace(".wav", ".pt")
            c_filename = c_filename.replace("DUMMY", "dataset/wavlm")
            c = torch.load(c_filename).squeeze(0)
        else:
            i = random.randint(68, 92)
            """
            basename = os.path.basename(filename)[:-4]
            spkname = basename[:4]
            #print(basename, spkname)
            with h5py.File(f"dataset/rs/wavlm/{spkname}/{i}.hdf5","r") as f:
                c = torch.from_numpy(f[basename][()]).squeeze(0)
            #print(c)
            """
            c_filename = filename.replace(".wav", f"_{i}.pt")
            c_filename = c_filename.replace("DUMMY", "dataset/sr/wavlm")
            c = torch.load(c_filename).squeeze(0)

        # 2023.01.10 update: code below can deteriorate model performance
        # I added these code during cleaning up, thinking that it can offer better performance than my
        # provided checkpoints, but actually it does the opposite.
        # What an act of 'adding legs to a snake'!
        """
        lmin = min(c.size(-1), spec.size(-1))
        spec, c = spec[:, :lmin], c[:, :lmin]
        audio_norm = audio_norm[:, :lmin*self.hop_length]
        _spec, _c, _audio_norm = spec, c, audio_norm
        while spec.size(-1) < self.spec_len:
            spec = torch.cat((spec, _spec), -1)
            c = torch.cat((c, _c), -1)
            audio_norm = torch.cat((audio_norm, _audio_norm), -1)
        start = random.randint(0, spec.size(-1) - self.spec_len)
        end = start + self.spec_len
        spec = spec[:, start:end]
        c = c[:, start:end]
        audio_norm = audio_norm[:, start*self.hop_length:end*self.hop_length]
        """

        if self.use_spk:
            return c, spec, audio_norm, filename, spk
        else:
            return c, spec, audio_norm, filename

    def __getitem__(self, index):
        data = self.get_audio(self.audiopaths[index][0])
        return data

    def __len__(self):
        return len(self.audiopaths)
