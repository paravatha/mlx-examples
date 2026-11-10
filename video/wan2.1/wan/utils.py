# Copyright © 2026 Apple Inc.

"""
Utility functions for Wan2.1 pipeline.

Weight loading, HF Hub downloading, video saving.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import numpy as np

from .model import WanModel
from .t5 import T5Encoder, create_umt5_xxl_encoder
from .tokenizers import T5Tokenizer
from .vae import WanVAE


@dataclass
class ModelSpec:
    repo_id: str
    repo_dit: str
    repo_vae: str
    repo_t5: str
    dit_params: Dict
    repo_tokenizer: str = "google/umt5-xxl/tokenizer.json"
    repo_clip: Optional[str] = None
    ckpt_path: Optional[str] = None
    repo_dit_2: Optional[str] = None
    is_moe: bool = False
    boundary_ratio: Optional[float] = None
    vae_params: Optional[Dict] = None
    vae_repo_id: Optional[str] = None
    t5_repo_id: Optional[str] = None
    tokenizer_repo_id: Optional[str] = None


spec_t2v_a14b = ModelSpec(
    repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    repo_dit="transformer/diffusion_pytorch_model.safetensors.index.json",
    repo_dit_2="transformer_2/diffusion_pytorch_model.safetensors.index.json",
    repo_vae="Wan2.1_VAE.pth",
    vae_repo_id="Wan-AI/Wan2.1-T2V-14B",
    repo_t5="models_t5_umt5-xxl-enc-bf16.pth",
    t5_repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    is_moe=True,
    boundary_ratio=0.875,
    dit_params={
        "dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
        "in_dim": 16,
        "out_dim": 16,
    },
    vae_params={"z_dim": 16, "dim": 96},
    ckpt_path=os.getenv("WAN_T2V_A14B"),
)

spec_ti2v_5b = ModelSpec(
    repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    repo_dit="transformer/diffusion_pytorch_model.safetensors.index.json",
    repo_vae="vae/diffusion_pytorch_model.safetensors",
    repo_t5="models_t5_umt5-xxl-enc-bf16.pth",
    t5_repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    is_moe=False,
    dit_params={
        "dim": 3072,
        "ffn_dim": 14336,
        "num_heads": 24,
        "num_layers": 30,
        "in_dim": 48,
        "out_dim": 48,
    },
    vae_params={
        "z_dim": 48,
        "dim": 160,
        "decoder_dim": 256,
        "in_channels": 12,
        "out_channels": 12,
        "is_residual": True,
    },
    ckpt_path=os.getenv("WAN_TI2V_5B"),
)

configs = {
    "t2v-1.3B": ModelSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        repo_dit="diffusion_pytorch_model.safetensors",
        repo_vae="Wan2.1_VAE.pth",
        repo_t5="models_t5_umt5-xxl-enc-bf16.pth",
        dit_params={"dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30},
        ckpt_path=os.getenv("WAN_T2V_1_3B"),
    ),
    "t2v-14B": ModelSpec(
        repo_id="Wan-AI/Wan2.1-T2V-14B",
        repo_dit="diffusion_pytorch_model.safetensors.index.json",
        repo_vae="Wan2.1_VAE.pth",
        repo_t5="models_t5_umt5-xxl-enc-bf16.pth",
        dit_params={"dim": 5120, "ffn_dim": 13824, "num_heads": 40, "num_layers": 40},
        ckpt_path=os.getenv("WAN_T2V_14B"),
    ),
    "i2v-14B": ModelSpec(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
        repo_dit="diffusion_pytorch_model.safetensors.index.json",
        repo_vae="Wan2.1_VAE.pth",
        repo_t5="models_t5_umt5-xxl-enc-bf16.pth",
        repo_clip="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        dit_params={
            "dim": 5120,
            "ffn_dim": 13824,
            "num_heads": 40,
            "num_layers": 40,
            "model_type": "i2v",
            "in_dim": 36,
        },
        ckpt_path=os.getenv("WAN_I2V_14B"),
    ),
    # Wan 2.2 TI2V-5B
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": spec_ti2v_5b,
    "Wan2.2-TI2V-5B": spec_ti2v_5b,
    "ti2v-5B": spec_ti2v_5b,
    # Wan 2.2 T2V-A14B (MoE)
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": spec_t2v_a14b,
    "Wan2.2-T2V-A14B": spec_t2v_a14b,
    "t2v-A14B": spec_t2v_a14b,
}


def _hf_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _load_weights(path: str) -> dict:
    """Unified loader for safetensors, sharded index, and .pth files."""
    assert os.path.isfile(path), f"Weights file at {path} does not exist"
    if path.endswith(".index.json"):
        weight_dir = os.path.dirname(path)
        with open(path) as f:
            index = json.load(f)
        weight_files = set(index["weight_map"].values())
        # Ensure all shards are downloaded (for HF Hub paths)
        for wf in weight_files:
            shard_path = os.path.join(weight_dir, wf)
            if not os.path.exists(shard_path):
                # Infer repo_id from HF cache path structure
                # .../models--Org--Repo/snapshots/hash/file
                parts = Path(path).parts
                for i, p in enumerate(parts):
                    if p.startswith("models--"):
                        repo_id = p.replace("models--", "").replace("--", "/")
                        snapshot_idx = -1
                        for j in range(i, len(parts)):
                            if parts[j] == "snapshots" and j + 1 < len(parts):
                                snapshot_idx = j + 1
                                break
                        if snapshot_idx != -1 and snapshot_idx < len(parts) - 1:
                            subfolder = "/".join(parts[snapshot_idx + 1 : -1])
                            hf_filename = f"{subfolder}/{wf}" if subfolder else wf
                        else:
                            hf_filename = wf
                        _hf_download(repo_id, hf_filename)
                        break
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(os.path.join(weight_dir, wf)))
        return weights
    elif path.endswith(".pth"):
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=True)
        weights = {k: mx.array(v.float().numpy()) for k, v in sd.items()}

        del torch

        return weights
    else:
        return mx.load(path)


def load_dit(
    name: str, checkpoint: Optional[str] = None, expert: Optional[str] = None
) -> WanModel:
    """Load DiT model with weights from HF Hub."""
    spec = configs[name]
    model = WanModel(**spec.dit_params)
    ckpt_path = checkpoint or spec.ckpt_path
    if ckpt_path is None:
        target_dit = (
            spec.repo_dit_2 if (expert == "low" and spec.repo_dit_2) else spec.repo_dit
        )
        try:
            ckpt_path = _hf_download(spec.repo_id, target_dit)
        except Exception:
            # Fallback if single safetensors vs sharded index.json
            if target_dit.endswith(".index.json"):
                alt_dit = target_dit.replace(".index.json", "")
                ckpt_path = _hf_download(spec.repo_id, alt_dit)
            elif target_dit.endswith(".safetensors"):
                alt_dit = f"{target_dit}.index.json"
                ckpt_path = _hf_download(spec.repo_id, alt_dit)
            else:
                raise
    weights = _load_weights(ckpt_path)
    weights = WanModel.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    return model


def load_vae(name: str) -> WanVAE:
    """Load VAE decoder with weights from HF Hub."""
    spec = configs[name]
    vae_params = spec.vae_params or {}
    vae = WanVAE(**vae_params)
    repo = spec.vae_repo_id or spec.repo_id
    try:
        ckpt_path = _hf_download(repo, spec.repo_vae)
    except Exception:
        if spec.repo_vae.endswith(".safetensors"):
            # Fallback to .pth or Wan2.1_VAE.pth
            ckpt_path = _hf_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
        else:
            raise
    weights = _load_weights(ckpt_path)
    weights = WanVAE.sanitize(weights)
    vae.load_weights(list(weights.items()), strict=False)
    return vae


def load_t5(name: str) -> T5Encoder:
    """Load T5 encoder with weights from HF Hub."""
    spec = configs[name]
    t5 = create_umt5_xxl_encoder()
    repo = spec.t5_repo_id or spec.repo_id
    weight_path = _hf_download(repo, spec.repo_t5)
    weights = _load_weights(weight_path)
    weights = T5Encoder.sanitize(weights)
    t5.load_weights(list(weights.items()), strict=True)
    return t5


def load_clip(name: str):
    """Load CLIP vision encoder with weights from HF Hub."""
    from .clip import CLIPVisionEncoder

    spec = configs[name]
    if spec.repo_clip is None:
        raise ValueError(f"Model {name} does not have a CLIP config")
    clip = CLIPVisionEncoder()
    weight_path = _hf_download(spec.repo_id, spec.repo_clip)
    weights = _load_weights(weight_path)
    weights = CLIPVisionEncoder.sanitize(weights)
    clip.load_weights(list(weights.items()), strict=True)
    return clip


def load_t5_tokenizer(name: str) -> T5Tokenizer:
    """Load T5 tokenizer from HF Hub."""
    spec = configs[name]
    repo = spec.tokenizer_repo_id or spec.repo_id
    tok_path = _hf_download(repo, spec.repo_tokenizer)
    return T5Tokenizer(tok_path)


def save_video(
    frames: mx.array,
    output_path: str,
    fps: int = 16,
) -> bool:
    """
    Save video frames to file using ffmpeg.

    Args:
        frames: Video tensor [T, H, W, C] (channels-last) in [-1, 1] or [0, 1]
        output_path: Output file path
        fps: Frames per second

    Returns:
        True if successful
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim == 5:
        frames = frames[0]

    # Convert from [-1, 1] to [0, 1]
    mx.eval(frames)
    if frames.min().item() < 0:
        frames = (frames + 1.0) / 2.0

    frames_np = np.array(mx.clip(frames * 255, 0, 255).astype(mx.uint8))
    T, H, W, C = frames_np.shape

    print(f"Saving {T} frames ({W}x{H}) to {output_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "fast",
        output_path,
    ]

    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate(input=frames_np.tobytes())
        if process.returncode != 0:
            print(f"FFmpeg error: {stderr.decode()}")
            return False
        print(f"Video saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        return False
