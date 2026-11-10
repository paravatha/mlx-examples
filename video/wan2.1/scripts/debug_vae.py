"""Debug script to check VAE decode output for white screen issues."""
import sys
sys.path.insert(0, "/Users/kal/git-ws/local-ai/mlx-examples/video/wan2.1")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from wan.utils import configs, load_vae, load_dit
from wan.vae import WanVAE, unpatchify_3d_channels_last, patchify_3d_channels_last

model_name = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
spec = configs[model_name]

print("=" * 60)
print("DEBUG: Wan2.2 TI2V-5B VAE Decode Analysis")
print("=" * 60)

# 1. Check VAE config
print(f"\n[1] VAE config:")
print(f"  z_dim:          {spec.vae_params['z_dim']}")
print(f"  dim:            {spec.vae_params['dim']}")
print(f"  out_channels:   {spec.vae_params['out_channels']}")
print(f"  in_channels:    {spec.vae_params['in_channels']}")
print(f"  is_residual:    {spec.vae_params['is_residual']}")

# 2. Load VAE
print(f"\n[2] Loading VAE...")
vae = load_vae(model_name)
print(f"  vae.z_dim:         {vae.z_dim}")
print(f"  vae.out_channels:  {vae.out_channels}")
print(f"  vae.mean shape:    {vae.mean.shape}")
print(f"  vae.std shape:     {vae.std.shape}")
print(f"  vae.mean[:5]:      {vae.mean[:5].tolist()}")
print(f"  vae.std[:5]:       {vae.std[:5].tolist()}")

# 3. Check unpatchify vs pixel_shuffle
print(f"\n[3] Testing unpatchify_3d_channels_last:")
test_in = mx.arange(1 * 1 * 2 * 2 * 12, dtype=mx.float32).reshape(1, 1, 2, 2, 12)
out_unp = unpatchify_3d_channels_last(test_in, 2)
print(f"  Input shape:   {test_in.shape}")
print(f"  Output shape:  {out_unp.shape}")
print(f"  Expected:      (1, 1, 4, 4, 3)")

# Round-trip test
test_rgb = mx.arange(1 * 1 * 4 * 4 * 3, dtype=mx.float32).reshape(1, 1, 4, 4, 3)
patched = patchify_3d_channels_last(test_rgb, 2)
unpatched = unpatchify_3d_channels_last(patched, 2)
roundtrip_ok = mx.allclose(test_rgb, unpatched).item()
print(f"  Patchify->Unpatchify roundtrip: {'PASS' if roundtrip_ok else 'FAIL'}")

# 4. Decode with random latents (simulating what the pipeline does)
print(f"\n[4] Decoding random latents (2 frames, 4x4 spatial)...")
z = mx.random.normal((2, 4, 4, 48))  # small test
print(f"  Latent shape: {z.shape}")
print(f"  Latent stats: min={z.min().item():.3f} max={z.max().item():.3f} mean={z.mean().item():.3f}")

# Step through decode manually
z_batch = z[None]  # [1, F, H, W, C]
print(f"  z_batch shape: {z_batch.shape}")

# Unscale
scale = 1.0 / vae.std
z_unscaled = z_batch / scale.reshape(1, 1, 1, 1, vae.z_dim) + vae.mean.reshape(1, 1, 1, 1, vae.z_dim)
mx.eval(z_unscaled)
print(f"  After unscale: min={z_unscaled.min().item():.3f} max={z_unscaled.max().item():.3f} mean={z_unscaled.mean().item():.3f}")

# Pre-decoder conv
x = vae.conv2(z_unscaled)
mx.eval(x)
print(f"  After conv2: shape={x.shape} min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f}")

# Decode frame-by-frame
num_frames = x.shape[1]
feat_cache = [None] * vae.decoder.num_cache_slots
outputs = []
for i in range(num_frames):
    frame = x[:, i:i+1, :, :, :]
    out_frame, feat_cache = vae._compiled_decode(frame, feat_cache)
    mx.eval(out_frame)
    outputs.append(out_frame)
    print(f"  Frame {i}: shape={out_frame.shape} min={out_frame.min().item():.3f} max={out_frame.max().item():.3f} mean={out_frame.mean().item():.3f}")

out = mx.concatenate(outputs, axis=1)
print(f"  Concatenated: shape={out.shape}")

if vae.out_channels == 12:
    print(f"  Applying unpatchify (out_channels=12)...")
    out = unpatchify_3d_channels_last(out, 2)
    print(f"  After unpatchify: shape={out.shape}")

mx.eval(out)
print(f"  Before clip: min={out.min().item():.3f} max={out.max().item():.3f} mean={out.mean().item():.3f}")

out = mx.clip(out, -1.0, 1.0)
mx.eval(out)
print(f"  After clip:  min={out.min().item():.3f} max={out.max().item():.3f} mean={out.mean().item():.3f}")

# Convert to uint8 like save_video does
video = out[0]  # remove batch dim
print(f"  Video shape: {video.shape}")

if video.min().item() < 0:
    video_01 = (video + 1.0) / 2.0
else:
    video_01 = video

video_uint8 = mx.clip(video_01 * 255, 0, 255).astype(mx.uint8)
mx.eval(video_uint8)
arr = np.array(video_uint8)
print(f"\n[5] Final uint8 video stats:")
print(f"  Shape: {arr.shape}")
print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.1f}, Std: {arr.std():.1f}")

# Check if it's basically all white (>240)
white_frac = (arr > 240).mean()
print(f"  Fraction > 240: {white_frac:.3f}")
if white_frac > 0.8:
    print("  >>> WARNING: Output is mostly WHITE! VAE decode is broken.")
elif white_frac > 0.5:
    print("  >>> WARNING: Output is very bright. Possible issue.")
else:
    print("  >>> Output has good dynamic range.")

print(f"\n  Sample pixel [0, 0, 0]: {arr[0, 0, 0]}")
print(f"  Sample pixel [0, H//2, W//2]: {arr[0, arr.shape[1]//2, arr.shape[2]//2]}")
print(f"\nDone.")

