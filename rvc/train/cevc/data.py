"""Dataset extensions for CEVC adapter-only training."""

from __future__ import annotations

import os

import numpy as np
import torch

from rvc.train.data_utils import TextAudioLoaderMultiNSFsid


class CEVCTextAudioLoader(TextAudioLoaderMultiNSFsid):
    def __init__(self, hparams, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.expressive_dir = os.path.join(experiment_dir, "expressive")
        self.roughness_dir = os.path.join(experiment_dir, "roughness")
        super().__init__(hparams)

    def _load_cevc_features(self, audio_path: str, target_length: int):
        basename = os.path.basename(audio_path)
        expressive_path = os.path.join(self.expressive_dir, basename + ".npy")
        roughness_path = os.path.join(self.roughness_dir, basename + ".npy")
        if not os.path.exists(expressive_path) or not os.path.exists(roughness_path):
            raise FileNotFoundError(
                f"Missing CEVC features for {basename}. Run Extract Features with "
                "CEVC expressive features enabled."
            )

        expressive = np.load(expressive_path, allow_pickle=False).astype(np.float32)
        roughness = np.load(roughness_path, allow_pickle=False).astype(np.float32)
        if expressive.ndim != 2:
            raise ValueError(f"Invalid expressive feature shape for {basename}")
        length = min(target_length, expressive.shape[0], roughness.shape[0])
        if length <= 0:
            raise ValueError(f"Empty CEVC features for {basename}")

        expressive = torch.from_numpy(expressive[:length]).transpose(0, 1)
        roughness = torch.from_numpy(roughness[:length])
        if length < target_length:
            pad = target_length - length
            expressive = torch.nn.functional.pad(
                expressive.unsqueeze(0), (0, pad), mode="replicate"
            ).squeeze(0)
            roughness = torch.nn.functional.pad(
                roughness[None, None, :], (0, pad), mode="replicate"
            ).view(-1)
        return expressive, roughness

    def get_audio_text_pair(self, audiopath_and_text):
        audio_path = audiopath_and_text[0]
        spec, wav, phone, pitch, pitchf, speaker_id = super().get_audio_text_pair(
            audiopath_and_text
        )
        expressive, roughness = self._load_cevc_features(
            audio_path, target_length=phone.shape[0]
        )
        return (
            spec,
            wav,
            phone,
            pitch,
            pitchf,
            speaker_id,
            expressive,
            roughness,
        )


class CEVCTextAudioCollate:
    def __call__(self, batch):
        _, order = torch.sort(
            torch.LongTensor([row[0].size(1) for row in batch]),
            dim=0,
            descending=True,
        )
        batch_size = len(batch)
        max_spec = max(row[0].size(1) for row in batch)
        max_wave = max(row[1].size(1) for row in batch)
        max_phone = max(row[2].size(0) for row in batch)
        feature_dim = batch[0][6].size(0)

        spec = torch.zeros(batch_size, batch[0][0].size(0), max_spec)
        spec_lengths = torch.zeros(batch_size, dtype=torch.long)
        wave = torch.zeros(batch_size, 1, max_wave)
        wave_lengths = torch.zeros(batch_size, dtype=torch.long)
        phone = torch.zeros(batch_size, max_phone, batch[0][2].size(1))
        phone_lengths = torch.zeros(batch_size, dtype=torch.long)
        pitch = torch.zeros(batch_size, max_phone, dtype=torch.long)
        pitchf = torch.zeros(batch_size, max_phone)
        speaker_id = torch.zeros(batch_size, dtype=torch.long)
        expressive = torch.zeros(batch_size, feature_dim, max_phone)
        roughness = torch.zeros(batch_size, max_phone)

        for destination, source_index in enumerate(order.tolist()):
            row = batch[source_index]
            spec[destination, :, : row[0].size(1)] = row[0]
            spec_lengths[destination] = row[0].size(1)
            wave[destination, :, : row[1].size(1)] = row[1]
            wave_lengths[destination] = row[1].size(1)
            phone[destination, : row[2].size(0)] = row[2]
            phone_lengths[destination] = row[2].size(0)
            pitch[destination, : row[3].size(0)] = row[3]
            pitchf[destination, : row[4].size(0)] = row[4]
            speaker_id[destination] = row[5]
            expressive[destination, :, : row[6].size(1)] = row[6]
            roughness[destination, : row[7].size(0)] = row[7]

        return (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
            speaker_id,
            expressive,
            roughness,
        )
