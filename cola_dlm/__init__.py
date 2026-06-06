# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cola DLM — Continuous Latent Diffusion Language Model.

``cola_dlm`` is the official HuggingFace Transformers-compatible
open-source release of the paper *Continuous Latent Diffusion Language
Model* (arXiv:2605.06548, https://arxiv.org/abs/2605.06548).

Cola DLM is a **hierarchical latent-variable language model** with the
joint factorization (Eq. 2.1.1 of the paper)::

    p(x, z_0) = p_theta(x | z_0) * p_psi(z_0),
    p(x)      = ∫ p_theta(x | z_0) * p_psi(z_0) dz_0,

implemented by two cooperating modules and an inference encoder:

* :class:`ColaTextVAEModel` — a Text VAE that contains both the
  inference encoder ``q_phi(z_0 | x)`` (used at training and at
  prefix-encoding time) and the conditional decoder
  ``p_theta(x | z_0)`` (used to realize discrete tokens from a latent
  sequence).
* :class:`ColaDiTModel` — a block-causal 1-D Diffusion Transformer
  parameterizing the latent prior ``p_psi(z_0)``. Training fits the
  vector field ``v_psi(z_t, t; z_0^(<b))`` via conditional Flow
  Matching (Eq. 2.1.7) under the visible set
  ``V_b = {sg(z_0^(<b)), z_t^(b)}`` (Eq. 2.2.3); inference realizes one
  block of the integrator ``Phi^psi_{0<-1}`` (Eq. 2.1.2 / 2.2.5) per
  step.

From a unified Markov-path perspective, the diffusion process performs
**latent prior transport** rather than token-level observation
recovery, which is the central distinction between Cola DLM and AR /
LLaDA / Plaid (Section 2.3 of the paper). See the project README,
``docs/architecture.md`` and the project page
https://hongcanguo.github.io/Cola-DLM/ for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .attention_utils import (
    create_2l_block_causal_mask,
    create_na_block_causal_mask,
    cu_seqlens,
    get_seqlen,
    max_seqlen,
)
from .configuration_cola_dit import ColaDiTConfig
from .configuration_cola_vae import ColaTextVAEConfig
from .modeling_cola_dit import ColaDiTModel
from .modeling_cola_vae import ColaTextVAEModel

__version__ = "0.1.0"

# Inference helpers are resolved lazily so that
#   ``python -m cola_dlm.inference``
# does not import the module twice (which would otherwise trigger the
# `RuntimeWarning: 'cola_dlm.inference' found in sys.modules ...` from runpy).
_LAZY_INFERENCE_ATTRS = {
    "apply_prompt_template",
    "generate_task_repaint_inference",
    "sample_with_strategies",
}

if TYPE_CHECKING:  # pragma: no cover - hints only, not executed
    from .inference import (  # noqa: F401
        apply_prompt_template,
        generate_task_repaint_inference,
        sample_with_strategies,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY_INFERENCE_ATTRS:
        from . import inference as _inference_mod

        return getattr(_inference_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_INFERENCE_ATTRS)


__all__ = [
    "ColaDiTConfig",
    "ColaDiTModel",
    "ColaTextVAEConfig",
    "ColaTextVAEModel",
    "apply_prompt_template",
    "create_2l_block_causal_mask",
    "create_na_block_causal_mask",
    "cu_seqlens",
    "generate_task_repaint_inference",
    "get_seqlen",
    "max_seqlen",
    "sample_with_strategies",
    "__version__",
]
