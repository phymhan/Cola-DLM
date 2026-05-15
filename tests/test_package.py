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

"""Package-level smoke tests.

These tests run without any GPU / model checkpoint and verify that the
public API surface is importable and exposes the expected symbols.
"""

from __future__ import annotations


def test_public_api_importable() -> None:
    import cola_dlm

    expected = {
        "ColaDiTConfig",
        "ColaDiTModel",
        "ColaTextVAEConfig",
        "ColaTextVAEModel",
        "create_na_block_causal_mask",
        "cu_seqlens",
        "get_seqlen",
        "max_seqlen",
    }
    missing = expected - set(dir(cola_dlm))
    assert not missing, f"Missing public API symbols: {sorted(missing)}"


def test_version_string() -> None:
    import cola_dlm

    assert isinstance(cola_dlm.__version__, str)
    # Basic Semver sanity: at least "MAJOR.MINOR.PATCH".
    parts = cola_dlm.__version__.split(".")
    assert len(parts) >= 3, cola_dlm.__version__
    for p in parts[:3]:
        # Each core segment must start with digits (e.g. "0", "1", "12rc1").
        assert p[0].isdigit(), cola_dlm.__version__


def test_configs_instantiate() -> None:
    """ColaDiTConfig / ColaTextVAEConfig should construct with their defaults."""

    from cola_dlm import ColaDiTConfig, ColaTextVAEConfig

    dit_cfg = ColaDiTConfig()
    vae_cfg = ColaTextVAEConfig()

    assert dit_cfg.model_type == "cola_dit"
    assert vae_cfg.model_type == "cola_text_vae"
    # Spot-check a few canonical defaults documented in the README.
    assert dit_cfg.emb_dim > 0
    assert vae_cfg.vocab_size > 0
