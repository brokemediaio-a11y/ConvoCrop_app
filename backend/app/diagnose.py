"""
Run this from the backend directory:
    python -m app.diagnose

It checks device placement, dtype, VRAM, and runs a quick
generation benchmark to isolate the bottleneck.
"""
import sys
import time
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("ConvoCrop Backend Diagnostics")
    print("=" * 60)

    # ── 1. CUDA check ──
    print(f"\n[1] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device name:    {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"    Total VRAM:     {total:.1f} GB")
        print(f"    CUDA version:   {torch.version.cuda}")
    else:
        print("    !! NO CUDA — model will run on CPU (very slow)")
        print("    Check your PyTorch installation: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    # ── 2. Load model ──
    print(f"\n[2] Loading model...")
    t0 = time.time()
    from app.model_loader import load_model
    model, tokenizer, image_processor, device = load_model()
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # ── 3. Check model type and device placement ──
    print(f"\n[3] Model diagnostics")
    print(f"    Model class: {type(model).__name__}")
    print(f"    Device var:   {device}")

    # Check where parameters actually live
    devices_found = set()
    dtypes_found = set()
    total_params = 0
    for name, param in model.named_parameters():
        devices_found.add(str(param.device))
        dtypes_found.add(str(param.dtype))
        total_params += param.numel()

    print(f"    Total params:  {total_params / 1e6:.1f}M")
    print(f"    Param devices: {devices_found}")
    print(f"    Param dtypes:  {dtypes_found}")

    if len(devices_found) > 1:
        print("\n    !! WARNING: Parameters on MULTIPLE devices!")
        print("    This causes CPU<->GPU transfers on every forward pass.")
        print("    Breakdown:")
        device_counts = {}
        for name, param in model.named_parameters():
            d = str(param.device)
            device_counts[d] = device_counts.get(d, 0) + param.numel()
        for d, count in sorted(device_counts.items()):
            print(f"      {d}: {count/1e6:.1f}M params")

    if "torch.float32" in dtypes_found and device.type == "cuda":
        print("\n    !! WARNING: Some params are float32 on CUDA!")
        print("    This doubles memory and halves speed.")
        dtype_counts = {}
        for name, param in model.named_parameters():
            d = str(param.dtype)
            dtype_counts[d] = dtype_counts.get(d, 0) + param.numel()
        for d, count in sorted(dtype_counts.items()):
            print(f"      {d}: {count/1e6:.1f}M params")

    # ── 4. VRAM usage ──
    print(f"\n[4] VRAM usage")
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"    Allocated: {allocated:.2f} GB")
    print(f"    Reserved:  {reserved:.2f} GB")

    # ── 5. Check if model has custom generate/forward ──
    print(f"\n[5] Model architecture check")
    has_custom_generate = hasattr(model, "generate")
    print(f"    Has generate():       {has_custom_generate}")
    has_vision_tower = hasattr(model, "get_vision_tower") or (
        hasattr(model, "model") and hasattr(model.model, "get_vision_tower")
    )
    print(f"    Has vision tower:     {has_vision_tower}")

    # Check if forward() accepts images kwarg
    import inspect
    fwd_sig = inspect.signature(model.forward)
    accepts_images = "images" in fwd_sig.parameters
    print(f"    forward(images=...):  {accepts_images}")
    if not accepts_images and hasattr(model, "model"):
        inner_sig = inspect.signature(model.model.forward)
        accepts_images_inner = "images" in inner_sig.parameters
        print(f"    model.model.forward(images=...): {accepts_images_inner}")

    # ── 6. Quick text-only generation benchmark ──
    print(f"\n[6] Text-only generation benchmark (no image)")
    input_text = "Hello, this is a test"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.inference_mode():
        _ = model.generate(
            inputs=input_ids,
            max_new_tokens=5,
            do_sample=False,
        )

    # Timed run
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(
            inputs=input_ids,
            max_new_tokens=32,
            do_sample=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    gen_tokens = out.shape[1] - input_ids.shape[1]
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    print(f"    Generated {gen_tokens} tokens in {elapsed:.2f}s")
    print(f"    Speed: {tps:.1f} tokens/sec")

    if tps < 5:
        print(f"\n    !! VERY SLOW: {tps:.1f} tok/s — expected 15-30 tok/s on 4050 Ti")
        print("    Likely causes:")
        print("      - Model params on wrong device (check [3] above)")
        print("      - Model in float32 instead of float16")
        print("      - CUDA not being used for computation")
    elif tps < 15:
        print(f"\n    ⚠ SLOW: {tps:.1f} tok/s — expected 15-30 tok/s")
    else:
        print(f"\n    ✓ GOOD: {tps:.1f} tok/s")

    # ── 7. Image processing benchmark ──
    print(f"\n[7] Image + generation benchmark")
    from PIL import Image
    import numpy as np
    # Create a dummy 768x768 green image (simulates a rice leaf photo)
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8)
    ).convert("RGB")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pixel_values = image_processor(
        images=dummy_img, return_tensors="pt"
    )["pixel_values"].to(device, dtype=dtype)

    print(f"    pixel_values shape: {pixel_values.shape}")
    print(f"    pixel_values device: {pixel_values.device}")
    print(f"    pixel_values dtype: {pixel_values.dtype}")

    # Build a simple prompt with image token
    from app.config import IMAGE_TOKEN_INDEX
    prompt = "<image>\nWhat disease does this rice leaf have?"
    pre, post = prompt.split("<image>", 1)
    pre_ids = tokenizer(pre, add_special_tokens=False, return_tensors="pt").input_ids
    post_ids = tokenizer(post, add_special_tokens=False, return_tensors="pt").input_ids
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)

    print(f"    input_ids shape: {input_ids.shape}")

    # Warmup
    with torch.inference_mode():
        try:
            _ = model.generate(
                inputs=input_ids,
                images=pixel_values,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            warmup_ok = True
        except Exception as e:
            print(f"    !! Warmup failed: {e}")
            warmup_ok = False

    if warmup_ok:
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model.generate(
                        inputs=input_ids,
                        images=pixel_values,
                        max_new_tokens=32,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                out = model.generate(
                    inputs=input_ids,
                    images=pixel_values,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

        gen_tokens = max(0, out.shape[1] - input_ids.shape[1])
        tps = gen_tokens / elapsed if elapsed > 0 else 0
        print(f"    Generated {gen_tokens} tokens in {elapsed:.2f}s")
        print(f"    Speed: {tps:.1f} tokens/sec (with image)")

        # VRAM after generation
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"    VRAM after generate: {allocated:.2f} GB")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Device:     {device}")
    print(f"  Param locs: {devices_found}")
    print(f"  Dtypes:     {dtypes_found}")
    print(f"  Model type: {type(model).__name__}")
    if warmup_ok:
        print(f"  Vision gen: {tps:.1f} tok/s")
    print()

    if len(devices_found) > 1:
        print("  → FIX: Parameters split across devices. See above.")
    elif "cpu" in str(devices_found).lower():
        print("  → FIX: Model is on CPU. Check CUDA installation.")
    elif "torch.float32" in dtypes_found:
        print("  → FIX: Model in float32. Set USE_HALF_PRECISION=true")
    elif warmup_ok and tps < 5:
        print("  → Vision tower may re-encode image every token.")
        print("    Consider pre-computing image embeddings once.")
    else:
        print("  → Model looks correctly configured.")


if __name__ == "__main__":
    main()