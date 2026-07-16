from typing import Optional

import torch

from rvc.lib.algorithm.commons import rand_slice_segments, slice_segments
from rvc.lib.algorithm.encoders import PosteriorEncoder, TextEncoder
from rvc.lib.algorithm.generators.registry import build_generator
from rvc.lib.algorithm.cevc.roughness_adapter import (
    DisabledRoughnessAdapter,
    RoughnessAdapter,
    RoughnessAdapterConfig,
)
from rvc.lib.algorithm.residuals import ResidualCouplingBlock


class Synthesizer(torch.nn.Module):
    """Base RVC synthesizer with generator selection delegated to a registry."""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        randomized: bool = True,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.randomized = randomized

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )

        print(f"Using {vocoder} vocoder")
        self.dec = build_generator(
            vocoder,
            use_f0=use_f0,
            initial_channel=inter_channels,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
            sample_rate=sr,
            checkpointing=checkpointing,
        )

        self.enc_q = PosteriorEncoder(
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
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)
        # The disabled adapter has no parameters, so legacy checkpoint keys stay exact.
        self.cevc_adapter = DisabledRoughnessAdapter()
        self.cevc_enabled = False
        self.cevc_inter_channels = inter_channels

    def enable_cevc_adapter(
        self,
        feature_dim: int = 5,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> RoughnessAdapter:
        config = RoughnessAdapterConfig(
            channels=self.cevc_inter_channels,
            feature_dim=feature_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.cevc_adapter = RoughnessAdapter(config)
        self.cevc_enabled = True
        return self.cevc_adapter

    def disable_cevc_adapter(self) -> None:
        self.cevc_adapter = DisabledRoughnessAdapter()
        self.cevc_enabled = False

    def _apply_cevc_adapter(
        self,
        latent: torch.Tensor,
        expressive_features: Optional[torch.Tensor],
        roughness: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.cevc_enabled or expressive_features is None or roughness is None:
            return latent
        return self.cevc_adapter(latent, expressive_features, roughness)

    def _remove_weight_norm_from(self, module):
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_lengths: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
        expressive_features: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            z_for_decoder = self._apply_cevc_adapter(
                z, expressive_features, roughness
            )
            if self.randomized:
                z_slice, ids_slice = rand_slice_segments(
                    z_for_decoder, y_lengths, self.segment_size
                )
                if self.use_f0:
                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    o = self.dec(z_slice, pitchf, g=g)
                else:
                    o = self.dec(z_slice, g=g)
                return o, ids_slice, x_mask, y_mask, (
                    z,
                    z_p,
                    m_p,
                    logs_p,
                    m_q,
                    logs_q,
                )

            if self.use_f0:
                o = self.dec(z_for_decoder, pitchf, g=g)
            else:
                o = self.dec(z_for_decoder, g=g)
            return o, None, x_mask, y_mask, (
                z,
                z_p,
                m_p,
                logs_p,
                m_q,
                logs_q,
            )

        return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
        expressive_features: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        z_for_decoder = self._apply_cevc_adapter(z, expressive_features, roughness)
        o = (
            self.dec(z_for_decoder * x_mask, nsff0, g=g)
            if self.use_f0
            else self.dec(z_for_decoder * x_mask, g=g)
        )
        return o, x_mask, (z, z_p, m_p, logs_p)
