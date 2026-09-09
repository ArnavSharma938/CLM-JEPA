"""ChemFM adaptation of official JaydenTeoh/NextLat horizon-1 training.

NextLat mechanics follow commit 3770be6009cea2b3c455a9ce7f2ca88b504bb955.
Only sequence masking and the native-vocabulary boundary are ChemFM-specific.
The dynamics model is training-only and never changes generation.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from decoder_projected import eligible_transitions
from stp import capture_final_hidden_state

NEXTLAT_REPOSITORY = "https://github.com/JaydenTeoh/NextLat"
NEXTLAT_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
NEXTLAT_CONFIG = "config/fineweb/1B/nextlat_finewebedu_1b_100b_horizon1.yaml"
PROJ_FACTOR = 1.6


class BiaslessLayerNorm(nn.Module):
    """Match upstream LayerNorm(..., bias=False)."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value, (value.shape[-1],), self.weight, None, 1e-5)


class NextLatDynamicsModel(nn.Module):
    """Direct architectural adaptation of upstream NextLatDynamicsModel."""

    def __init__(self, hidden_size: int, proj_factor: float = PROJ_FACTOR):
        super().__init__()
        input_dim = 2 * hidden_size
        hidden_dim = 128 * round(proj_factor * input_dim / 128)
        self.norm_x = BiaslessLayerNorm(input_dim)
        self.hidden_state_dropout = nn.Identity()  # official 1B config dropout=0
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_size, bias=False),
        )
        self.apply(self._init_weights)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, current_states: torch.Tensor, next_token_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.hidden_state_dropout(current_states)
        # Upstream order is [next-token embedding, current hidden state].
        inputs = self.norm_x(torch.cat([next_token_embeds, hidden_states], dim=-1))
        return current_states + self.mlp(inputs)


class FaithfulNextLatObjective(nn.Module):
    """Official horizon-1 NextLat objective with ChemFM product masking."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        product_start_token_id: int,
        eos_token_id: int,
        native_vocab_size: int,
        resume_state=None,
    ):
        super().__init__()
        hidden_size = int(weight.shape[1])
        self.predictor = NextLatDynamicsModel(hidden_size)
        self.product_start = int(product_start_token_id)
        self.eos = int(eos_token_id)
        self.native_vocab_size = int(native_vocab_size)
        self.alpha = 1.0  # compatibility field; no gradient calibration is applied
        self.calibration = None
        self.metadata = {
            "method": "faithful_nextlat_horizon1",
            "upstream_repository": NEXTLAT_REPOSITORY,
            "upstream_commit": NEXTLAT_COMMIT,
            "upstream_config": NEXTLAT_CONFIG,
            "hidden_size": hidden_size,
            "input_dim": self.predictor.input_dim,
            "proj_factor": PROJ_FACTOR,
            "predictor_hidden_dim": self.predictor.hidden_dim,
            "predictor_bias": False,
            "predictor_dropout": 0.0,
            "predictor_initialization": "normal_std_0.02_all_linear_weights",
            "lambda_mse": 1.0,
            "lambda_kl": 1.0,
            "lambda_ce": 0.0,
            "mtp_horizon": 1,
            "reduction": "global_valid_positions",
            "live_detached_lm_head": True,
            "current_hidden_gradient": True,
            "next_token_embedding_gradient": True,
            "future_hidden_stop_gradient": True,
            "chemfm_native_vocab_size": self.native_vocab_size,
            "chemfm_transition_mask": "actual product target tokens; first included; target EOS excluded",
            "initial_active_head_fp32_sha256": hashlib.sha256(
                weight[:native_vocab_size].detach().float().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }
        if resume_state is not None:
            saved = torch.load(resume_state, map_location="cpu", weights_only=False)
            # Preserve the immutable training-time metadata when loading a
            # frozen checkpoint. Compare every behavioral field known to the
            # current implementation; tolerate historical annotation-only
            # fields that do not occur in the module state or loss path.
            for name, value in self.metadata.items():
                if name != "initial_active_head_fp32_sha256" and name in saved["metadata"]:
                    if saved["metadata"][name] != value:
                        raise ValueError(f"faithful NextLat metadata mismatch: {name}")
            self.metadata = saved["metadata"]
            self.restore_training_state(resume_state)

    def losses(self, current, future, embedding, mask, live_head_weight):
        if not torch.any(mask):
            raise ValueError("empty transition batch")
        autocast = torch.autocast(
            device_type=current.device.type,
            dtype=torch.bfloat16,
            enabled=current.device.type == "cuda",
        )
        with autocast:
            predicted = self.predictor(current, embedding)
            target = future.detach()
            mse_element = F.smooth_l1_loss(predicted, target, reduction="none")
            weights = mask.unsqueeze(-1).to(mse_element.dtype)
            latent = (mse_element * weights).sum() / weights.expand_as(mse_element).sum().clamp_min(1)
            # Both distributions use the current active head. Detaching its weight
            # blocks auxiliary head updates while retaining gradients to predicted h.
            head = live_head_weight[: self.native_vocab_size].detach()
            teacher_logits = F.linear(target, head).detach()
            predicted_logits = F.linear(predicted, head)
            log_q = F.log_softmax(teacher_logits, dim=-1)
            log_p = F.log_softmax(predicted_logits, dim=-1)
            per_position_kl = F.kl_div(log_p, log_q, log_target=True, reduction="none").sum(-1)
            kl = (per_position_kl * mask.to(per_position_kl.dtype)).sum() / mask.sum().clamp_min(1)
        return latent.float(), kl.float(), predicted

    def forward(self, model, batch):
        mask = eligible_transitions(batch, self.product_start, self.eos)
        with capture_final_hidden_state(model) as captured:
            output = model(**{key: batch[key] for key in ("input_ids", "attention_mask", "labels")})
        # This separate lookup is mathematically the same embedding pathway as
        # upstream's captured token embeddings and deliberately retains gradient.
        embedding = model.get_input_embeddings()(batch["input_ids"][:, 1:])
        latent, kl, _ = self.losses(
            captured[0][:, :-1],
            captured[0][:, 1:],
            embedding,
            mask,
            model.get_output_embeddings().weight,
        )
        auxiliary = latent + kl
        return SimpleNamespace(
            loss=output.loss + auxiliary,
            native_loss=output.loss,
            logits=output.logits,
            jepa_loss=auxiliary,
            sigreg_loss=None,
            jepa_objective_loss=auxiliary,
            jepa_active=True,
            lz=latent,
            kl=kl,
        )

    def save_training_state(self, path):
        torch.save({"state_dict": self.state_dict(), "metadata": self.metadata}, path)

    def restore_training_state(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["metadata"] != self.metadata:
            raise ValueError("faithful NextLat metadata does not match resume configuration")
        self.load_state_dict(state["state_dict"])
