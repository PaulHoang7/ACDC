"""
Sanity check for all 3 ACDC segmentation models.

Tests per model:
    1. Instantiate (scratch + pretrained)
    2. Dummy forward pass → verify output shapes
    3. Loss computation → backward pass → no errors
    4. Parameter count + encoder/decoder breakdown

Usage:
    python scripts/test_models.py
    python scripts/test_models.py --model m1          # single model
    python scripts/test_models.py --device cuda       # GPU test
    python scripts/test_models.py --batch-size 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import build_model, MODEL_REGISTRY


def count_params(model: nn.Module) -> dict:
    """Count total, trainable, and per-component parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n

    return {"total": total, "trainable": trainable, "breakdown": breakdown}


def fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def test_model(model_name: str, device: str, batch_size: int) -> bool:
    """Run full sanity check on one model. Returns True if all pass."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    ok = True
    H, W = 256, 256
    C_in, C_out = 1, 4

    for init_mode in ["scratch", "pretrained"]:
        pretrained = (init_mode == "pretrained")
        tag = f"[{model_name} | {init_mode}]"
        print(f"\n{tag}")

        # ── 1. Instantiate ───────────────────────────────────────
        try:
            model = build_model(
                model_name,
                in_channels=C_in,
                num_classes=C_out,
                pretrained=pretrained,
            )
            model = model.to(device)
            model.train()
            print(f"  [1] Instantiate: OK")
        except Exception as e:
            print(f"  [1] Instantiate: FAIL — {e}")
            ok = False
            continue

        # ── 2. Forward pass ──────────────────────────────────────
        try:
            x = torch.randn(batch_size, C_in, H, W, device=device)
            out = model(x)

            # Check seg_logits
            seg = out["seg_logits"]
            assert seg.shape == (batch_size, C_out, H, W), \
                f"seg_logits shape {seg.shape} != ({batch_size},{C_out},{H},{W})"
            assert not torch.isnan(seg).any(), "NaN in seg_logits"
            assert not torch.isinf(seg).any(), "Inf in seg_logits"

            shape_report = f"seg_logits={list(seg.shape)}"

            # Check boundary_logits (M3 only)
            if "boundary_logits" in out:
                bnd = out["boundary_logits"]
                assert bnd.shape == (batch_size, 1, H, W), \
                    f"boundary shape {bnd.shape}"
                shape_report += f", boundary={list(bnd.shape)}"

            # Check aux_seg_logits (M3 only)
            if "aux_seg_logits" in out:
                for i, aux in enumerate(out["aux_seg_logits"]):
                    assert aux.shape == (batch_size, C_out, H, W), \
                        f"aux[{i}] shape {aux.shape}"
                shape_report += f", aux_heads={len(out['aux_seg_logits'])}"

            print(f"  [2] Forward:     OK — {shape_report}")

        except Exception as e:
            print(f"  [2] Forward:     FAIL — {e}")
            ok = False
            continue

        # ── 3. Backward pass ─────────────────────────────────────
        try:
            model.zero_grad()
            target = torch.randint(0, C_out, (batch_size, H, W), device=device)

            # Main segmentation loss
            loss = F.cross_entropy(out["seg_logits"], target)

            # Boundary loss (M3)
            if "boundary_logits" in out:
                bnd_target = torch.zeros(batch_size, 1, H, W, device=device)
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(
                    out["boundary_logits"], bnd_target
                )

            # Deep supervision loss (M3)
            if "aux_seg_logits" in out:
                for aux in out["aux_seg_logits"]:
                    loss = loss + 0.1 * F.cross_entropy(aux, target)

            loss.backward()

            # Check gradients exist
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters() if p.requires_grad
            )
            assert has_grad, "No gradients computed"

            # Check no NaN gradients
            nan_grad = any(
                torch.isnan(p.grad).any()
                for p in model.parameters()
                if p.grad is not None
            )
            assert not nan_grad, "NaN in gradients"

            print(f"  [3] Backward:    OK — loss={loss.item():.4f}")

        except Exception as e:
            print(f"  [3] Backward:    FAIL — {e}")
            ok = False
            continue

        # ── 4. Parameter count (only once per model) ─────────────
        if init_mode == "scratch":
            info = count_params(model)
            print(f"  [4] Params:      {fmt_params(info['total'])} total, "
                  f"{fmt_params(info['trainable'])} trainable")
            for comp, n in info["breakdown"].items():
                print(f"       {comp:20s} {fmt_params(n):>8s}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Model sanity check")
    parser.add_argument("--model", default=None,
                        help="Test single model (e.g. m1_unetr34)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print("ACDC Model Sanity Check")
    print(f"  device: {args.device}")
    print(f"  batch:  {args.batch_size}")

    if args.model:
        models_to_test = [args.model]
    else:
        models_to_test = list(MODEL_REGISTRY.keys())

    all_ok = True
    for name in models_to_test:
        passed = test_model(name, args.device, args.batch_size)
        if not passed:
            all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("ALL MODELS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
