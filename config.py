"""Typed model/training configuration loaded from YAML.

The YAML has two sections that map onto the two dataclasses below::

    model:   # architecture — must match a checkpoint
      num_slots: 4
      ...
    train:   # optimization — only used by train.py
      lr_enc: 1.0e-4
      ...

``ModelConfig`` is passed straight into the model constructors in place of the
old argparse ``Namespace``; attribute access (``cfg.num_slots``) is unchanged,
so ``sysbinder.py`` / ``dvae.py`` need no modifications. Runtime concerns
(paths, seed, wandb, topology flags) stay as CLI arguments.
"""

from dataclasses import dataclass, asdict, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters — these define the network shape and must
    match whatever a checkpoint was trained with."""

    image_size: int = 128
    image_channels: int = 3

    num_iterations: int = 3
    num_slots: int = 4
    num_blocks: int = 8
    cnn_hidden_size: int = 512
    slot_size: int = 2048
    mlp_hidden_size: int = 192
    num_prototypes: int = 64
    num_retrieval_iters: int = 1  # Hopfield retrieval steps in BlockPrototypeMemory (1 == single attention read)
    beta: float = 1.0  # MHN inverse temperature; >1 sharper retrieval, <1 more blending
    block_norm: str = 'layer'  # 'layer' = BlockLayerNorm, 'sphere' = per-block L2 norm

    vocab_size: int = 4096
    num_decoder_layers: int = 8
    num_decoder_heads: int = 4
    d_model: int = 192
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Optimization hyperparameters — only consumed by train.py."""

    batch_size: int = 40
    epochs: int = 500

    lr_dvae: float = 3e-4
    lr_enc: float = 1e-4
    lr_dec: float = 3e-4
    lr_warmup_steps: int = 30000
    lr_half_life: int = 250000
    clip: float = 0.05

    tau_start: float = 1.0
    tau_final: float = 0.1
    tau_steps: int = 30000

    use_dp: bool = True

    sigreg_weight: float = 0.0
    sigreg_num_slices: int = 256


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig


def _build(cls, section: dict[str, Any], name: str):
    """Instantiate a dataclass from a YAML section, rejecting unknown keys so a
    typo in the YAML fails loudly instead of being silently ignored."""
    section = section or {}
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{name}' config section: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**section)


def load_config(path: str) -> Config:
    """Load a Config from a YAML file with ``model`` and ``train`` sections."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"model", "train"}
    if unknown:
        raise ValueError(
            f"Unknown top-level key(s) in {path}: {sorted(unknown)}. "
            f"Expected 'model' and/or 'train'."
        )
    return Config(
        model=_build(ModelConfig, raw.get("model"), "model"),
        train=_build(TrainConfig, raw.get("train"), "train"),
    )


def save_config(cfg: Config, path: str) -> None:
    """Write a Config back to YAML, preserving the two-section layout."""
    with open(path, "w") as f:
        yaml.safe_dump(
            {"model": asdict(cfg.model), "train": asdict(cfg.train)},
            f,
            sort_keys=False,
        )


def flat_dict(cfg: Config) -> dict[str, Any]:
    """Flatten to a single dict for wandb logging."""
    return {**asdict(cfg.model), **asdict(cfg.train)}
