#!/usr/bin/env python3


try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise SystemExit(
        "PyTorch is not installed for this Python.\n"
        "Install dependencies with: python3 -m pip install -r requirements.txt"
    ) from error

try:
    import mlx.core as mx
except ModuleNotFoundError as error:
    if error.name != "mlx":
        raise
    raise SystemExit(
        "Apple MLX is not installed for this Python.\n"
        "Install dependencies with: python3 -m pip install -r requirements.txt"
    ) from error


def main():
    print("--- Apple Silicon Hardware Test ---")

    # Test PyTorch MPS (Metal Performance Shaders) availability.
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        _ = torch.ones(1, device=mps_device)
        print("✅ PyTorch is successfully utilizing the M5 GPU (MPS).")
    else:
        print("❌ PyTorch is NOT finding the MPS backend.")

    # Test MLX array generation.
    try:
        _ = mx.array([1.0, 2.0, 3.0])
        print("✅ Apple MLX is installed and functioning.")
    except Exception as error:
        print(f"❌ MLX encountered an error: {error}")


if __name__ == "__main__":
    main()
