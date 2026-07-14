"""One-latent CEVC A/B conversion used by the diagnostic tab."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import faiss
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal

from rvc.infer.cevc import CEVCPipeline, seed_cevc_inference
from rvc.infer.cevc_ab import (
    DEFAULT_AB_STRENGTHS,
    audio_metrics,
    normalize_strengths,
    validate_equal_lengths,
)
from rvc.infer.pipeline import AudioProcessor, ah, bh
from rvc.lib.utils import load_audio_infer


class OneLatentCEVCPipeline(CEVCPipeline):
    """Decode every roughness level from the same ContentVec/F0/latent state."""

    def voice_conversion_variants(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        strengths=DEFAULT_AB_STRENGTHS,
    ) -> dict[float, np.ndarray]:
        strengths = normalize_strengths(strengths)
        with torch.no_grad():
            pitch_guidance = pitch is not None and pitchf is not None
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            if feats.dim() != 1:
                raise ValueError(f"Expected mono input, received {feats.dim()} dimensions")
            feats = feats.view(1, -1).to(self.device)
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
            feats0 = feats.clone() if pitch_guidance else None

            if index:
                feats = self._retrieve_speaker_embeddings(
                    feats, index, big_npy, index_rate
                )
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            p_len_value = min(audio0.shape[0] // self.window, feats.shape[1])

            if pitch_guidance:
                feats0 = F.interpolate(
                    feats0.permute(0, 2, 1), scale_factor=2
                ).permute(0, 2, 1)
                pitch = pitch[:, :p_len_value]
                pitchf = pitchf[:, :p_len_value].float()
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None

            expressive_features, _ = self._build_conditioning(
                audio0, pitchf, feats.dtype
            )
            p_len = torch.tensor([p_len_value], device=self.device).long()
            g = net_g.emb_g(sid).unsqueeze(-1)
            m_p, logs_p, x_mask = net_g.enc_p(feats.float(), pitch, p_len)

            # This is the only stochastic latent sample for the whole A/B segment.
            z_p = (
                m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
            ) * x_mask
            z = net_g.flow(z_p, x_mask, g=g, reverse=True)

            outputs: dict[float, np.ndarray] = {}
            for strength in strengths:
                if strength <= 0.0:
                    z_for_decoder = z
                else:
                    control = torch.tensor(
                        [strength], device=z.device, dtype=z.dtype
                    )
                    z_for_decoder = net_g.cevc_adapter(
                        z, expressive_features.to(dtype=z.dtype), control
                    )
                generated = (
                    net_g.dec(z_for_decoder * x_mask, pitchf, g=g)
                    if net_g.use_f0
                    else net_g.dec(z_for_decoder * x_mask, g=g)
                )
                outputs[strength] = (
                    generated[0, 0].detach().cpu().float().numpy()
                )

            validate_equal_lengths(outputs)
            del feats, feats0, p_len, expressive_features, g, m_p, logs_p, x_mask, z_p, z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs

    def pipeline_variants(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        strengths=DEFAULT_AB_STRENGTHS,
    ) -> dict[float, np.ndarray]:
        strengths = normalize_strengths(strengths)
        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(f"An error occurred reading the FAISS index: {error}")
                index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        short_pad = np.pad(
            audio, (self.window // 2, self.window // 2), mode="reflect"
        )
        split_points = []
        if short_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for offset in range(self.window):
                audio_sum += short_pad[offset : offset - self.window]
            for center in range(self.t_center, audio.shape[0], self.t_center):
                window = np.abs(
                    audio_sum[center - self.t_query : center + self.t_query]
                )
                split_points.append(
                    center - self.t_query + np.where(window == window.min())[0][0]
                )

        padded = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = padded.shape[0] // self.window
        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance:
            pitch_values, pitchf_values = self.get_f0(
                padded, p_len, f0_method, pitch, False, 1.0, False, 155.0
            )
            pitch_values = pitch_values[:p_len]
            pitchf_values = pitchf_values[:p_len]
            if self.device == "mps":
                pitchf_values = pitchf_values.astype(np.float32)
            pitch_tensor = (
                torch.tensor(pitch_values, device=self.device).unsqueeze(0).long()
            )
            pitchf_tensor = (
                torch.tensor(pitchf_values, device=self.device).unsqueeze(0).float()
            )
        else:
            pitch_tensor = pitchf_tensor = None

        pieces: dict[float, list[np.ndarray]] = {value: [] for value in strengths}
        start = 0
        last_split = None

        def decode(segment, segment_pitch, segment_pitchf):
            generated = self.voice_conversion_variants(
                model,
                net_g,
                sid_tensor,
                segment,
                segment_pitch,
                segment_pitchf,
                index,
                big_npy,
                index_rate,
                version,
                protect,
                strengths,
            )
            for strength, values in generated.items():
                pieces[strength].append(
                    values[self.t_pad_tgt : -self.t_pad_tgt]
                )

        for split_at in split_points:
            split_at = split_at // self.window * self.window
            decode(
                padded[start : split_at + self.t_pad2 + self.window],
                (
                    pitch_tensor[
                        :, start // self.window : (split_at + self.t_pad2) // self.window
                    ]
                    if pitch_guidance
                    else None
                ),
                (
                    pitchf_tensor[
                        :, start // self.window : (split_at + self.t_pad2) // self.window
                    ]
                    if pitch_guidance
                    else None
                ),
            )
            start = split_at
            last_split = split_at

        decode(
            padded[last_split:],
            (
                pitch_tensor[:, last_split // self.window :]
                if last_split is not None
                else pitch_tensor
            ),
            (
                pitchf_tensor[:, last_split // self.window :]
                if last_split is not None
                else pitchf_tensor
            ),
        )

        outputs = {
            strength: np.concatenate(parts).astype(np.float32, copy=False)
            for strength, parts in pieces.items()
        }
        if volume_envelope != 1:
            outputs = {
                strength: AudioProcessor.change_rms(
                    audio, self.sample_rate, values, self.tgt_sr, volume_envelope
                ).astype(np.float32, copy=False)
                for strength, values in outputs.items()
            }
        validate_equal_lengths(outputs)
        return outputs


def convert_cevc_ab(
    converter,
    *,
    audio_input_path: str,
    output_paths: Mapping[float, str],
    report_path: str,
    index_path: str,
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    protect: float = 0.5,
    split_audio: bool = False,
    sid: int = 0,
) -> dict:
    """Generate one comparable A/B set and refuse truncated variants."""

    if split_audio:
        raise ValueError(
            "Split Audio is temporarily disabled in one-latent A/B mode. "
            "The internal RVC pipeline still segments long recordings safely."
        )

    base_pipeline = converter.vc
    converter.vc = OneLatentCEVCPipeline(
        converter.tgt_sr,
        converter.config,
        base_pipeline.feature_stats,
        1.0,
    )
    audio = load_audio_infer(audio_input_path, 16000)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio = audio / audio_max

    if not converter.hubert_model:
        converter.load_hubert("contentvec")
        converter.last_embedder_model = "contentvec"

    file_index = (
        str(index_path or "")
        .strip()
        .strip('"')
        .strip("\n")
        .strip()
        .replace("trained", "added")
    )
    seed_cevc_inference()
    outputs = converter.vc.pipeline_variants(
        model=converter.hubert_model,
        net_g=converter.net_g,
        sid=sid,
        audio=audio,
        pitch=int(pitch),
        f0_method=f0_method,
        file_index=file_index,
        index_rate=float(index_rate),
        pitch_guidance=converter.use_f0,
        volume_envelope=1.0,
        version=converter.version,
        protect=float(protect),
        strengths=output_paths.keys(),
    )

    sample_count = validate_equal_lengths(outputs)
    common_peak = max(float(np.max(np.abs(values))) for values in outputs.values())
    common_gain = 0.99 / common_peak if common_peak > 0.99 else 1.0
    if common_gain != 1.0:
        outputs = {
            strength: values * common_gain for strength, values in outputs.items()
        }

    output_report = {}
    for strength, destination in output_paths.items():
        values = outputs[float(strength)]
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, values, converter.tgt_sr, format="WAV")
        output_report[f"{float(strength):g}"] = {
            "path": destination,
            **audio_metrics(values, converter.tgt_sr),
        }

    report = {
        "format": "cevc-ab-report-v2",
        "shared_latent": True,
        "sample_count": sample_count,
        "sample_rate": int(converter.tgt_sr),
        "common_output_gain": common_gain,
        "input_path": audio_input_path,
        "index_path": file_index,
        "outputs": output_report,
    }
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
