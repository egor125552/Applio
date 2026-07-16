"""CEVC adapter loading and deterministic A/B inference support."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from rvc.infer.cevc_conditioning import (
    build_conditioning_tensors,
    find_cevc_adapter_for_model,
)
from rvc.infer.pipeline import Pipeline
from rvc.lib.algorithm.cevc.checkpoint import load_adapter_checkpoint


CEVC_AB_SEED = 20260714


def seed_cevc_inference(seed: int = CEVC_AB_SEED) -> None:
    """Reset every RNG used by the ordinary RVC inference path."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_cevc_adapter(model_path: str, adapter_path: Optional[str]) -> str:
    explicit = str(adapter_path or "").strip().strip('"')
    resolved = explicit or find_cevc_adapter_for_model(model_path)
    if not resolved:
        raise FileNotFoundError(
            "No CEVC adapter was selected and no unique *.cevc.pth file was "
            f"found beside the voice model: {model_path}"
        )
    resolved = os.path.abspath(os.path.expanduser(resolved))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"CEVC adapter not found: {resolved}")
    return resolved


def attach_cevc_adapter(voice_converter, adapter_path: str) -> dict:
    """Attach a standalone adapter checkpoint to an already loaded RVC network."""

    if voice_converter.net_g is None:
        raise RuntimeError("Load the base RVC model before attaching a CEVC adapter")

    adapter, payload = load_adapter_checkpoint(adapter_path, map_location="cpu")
    expected_sr = int(voice_converter.tgt_sr)
    checkpoint_sr = int(payload.get("sample_rate", 0))
    if checkpoint_sr != expected_sr:
        raise ValueError(
            f"CEVC sample rate {checkpoint_sr} does not match voice model {expected_sr}"
        )

    expected_channels = int(voice_converter.net_g.cevc_inter_channels)
    if int(adapter.config.channels) != expected_channels:
        raise ValueError(
            f"CEVC adapter channels {adapter.config.channels} do not match base "
            f"model channels {expected_channels}"
        )

    model_folder = Path(voice_converter.loaded_model or "").resolve().parent.name
    checkpoint_name = str(payload.get("model_name", ""))
    if checkpoint_name and model_folder and checkpoint_name != model_folder:
        print(
            "CEVC warning: adapter model_name "
            f"'{checkpoint_name}' differs from model folder '{model_folder}'.",
            flush=True,
        )

    adapter = adapter.to(voice_converter.config.device).float().eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    voice_converter.net_g.cevc_adapter = adapter
    voice_converter.net_g.cevc_enabled = True
    return payload


class CEVCPipeline(Pipeline):
    """Ordinary Applio pipeline with expressive features passed to the adapter."""

    def __init__(self, tgt_sr, config, feature_stats: dict, roughness_strength: float):
        super().__init__(tgt_sr, config)
        self.feature_stats = dict(feature_stats or {})
        self.roughness_strength = 0.0
        self.set_roughness_strength(roughness_strength)

    def set_roughness_strength(self, value: float) -> None:
        self.roughness_strength = float(np.clip(value, 0.0, 1.0))

    def _build_conditioning(self, audio0, pitchf, dtype):
        if pitchf is None:
            frame_count = max(1, int(np.ceil(len(audio0) / self.window)))
            f0 = np.zeros(frame_count, dtype=np.float32)
        else:
            f0 = pitchf.detach().float().cpu().numpy().reshape(-1)
        return build_conditioning_tensors(
            np.asarray(audio0, dtype=np.float32),
            f0,
            self.feature_stats,
            device=self.device,
            dtype=dtype,
            roughness_strength=self.roughness_strength,
        )

    def voice_conversion(
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
    ):
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

            expressive_features, roughness = self._build_conditioning(
                audio0, pitchf, feats.dtype
            )
            p_len = torch.tensor([p_len_value], device=self.device).long()
            audio1 = (
                net_g.infer(
                    feats.float(),
                    p_len,
                    pitch,
                    pitchf,
                    sid,
                    expressive_features=expressive_features,
                    roughness=roughness,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

            del feats, feats0, p_len, expressive_features, roughness
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio1


def prepare_cevc_converter(
    model_path: str,
    adapter_path: Optional[str],
    *,
    sid: int = 0,
    roughness_strength: float = 0.0,
):
    """Load a normal RVC model, attach its CEVC profile and replace its pipeline."""

    from rvc.infer.infer import VoiceConverter

    converter = VoiceConverter()
    converter.get_vc(model_path, sid)
    if converter.net_g is None or converter.vc is None:
        raise RuntimeError(f"Unable to load RVC voice model: {model_path}")

    resolved_adapter = resolve_cevc_adapter(model_path, adapter_path)
    payload = attach_cevc_adapter(converter, resolved_adapter)
    converter.vc = CEVCPipeline(
        converter.tgt_sr,
        converter.config,
        payload.get("feature_stats", {}),
        roughness_strength,
    )
    return converter, resolved_adapter, payload
