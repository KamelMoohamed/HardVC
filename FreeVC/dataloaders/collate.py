import torch

from commons import rand_spec_segments, slice_segments


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, train_use_sr, model_use_spk, data_hop_length, train_max_speclen):
        self.use_sr = train_use_sr
        self.use_spk = model_use_spk
        self.data_hop_length = data_hop_length
        self.train_max_speclen = train_max_speclen

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio, speaker identities, and filenames"""
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        filenames = []  # Collect filenames

        if self.use_spk:
            spks = torch.FloatTensor(len(batch), batch[4].size(0))
        else:
            spks = None

        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            filenames.append(row[3])  # Store filename

            c = row[0]
            c_padded[i, :, : c.size(1)] = c

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            if self.use_spk:
                spks[i] = row[4]

        spec_seglen = (
            spec_lengths[-1]
            if spec_lengths[-1] < self.train_max_speclen + 1
            else self.train_max_speclen + 1
        )
        wav_seglen = spec_seglen * self.data_hop_length

        spec_padded, ids_slice = rand_spec_segments(
            spec_padded, spec_lengths, spec_seglen
        )
        wav_padded = slice_segments(
            wav_padded, ids_slice * self.data_hop_length, wav_seglen
        )

        c_padded = slice_segments(c_padded, ids_slice, spec_seglen)[:, :, :-1]

        spec_padded = spec_padded[:, :, :-1]
        wav_padded = wav_padded[:, :, : -self.data_hop_length]

        if self.use_spk:
            return c_padded, spec_padded, wav_padded, spks, filenames
        else:
            return c_padded, spec_padded, wav_padded, filenames
