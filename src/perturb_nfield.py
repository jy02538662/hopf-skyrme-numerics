import argparse
from pathlib import Path

import numpy as np
import torch

from hopf_skyrme_torch import apply_boundary_, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a saved Ritz perturbation mode to a Hopf-Skyrme nfield")
    parser.add_argument("--field", required=True, help="Input nfield .npy")
    parser.add_argument("--mode", required=True, help="Input Ritz mode .npy")
    parser.add_argument("--eps", type=float, required=True, help="Perturbation amplitude")
    parser.add_argument("--out", required=True, help="Output nfield .npy")
    return parser.parse_args()


def main():
    args = parse_args()
    field = torch.as_tensor(np.load(args.field), dtype=torch.float32)
    mode = torch.as_tensor(np.load(args.mode), dtype=torch.float32)
    if field.shape != mode.shape:
        raise ValueError(f"field shape {field.shape} != mode shape {mode.shape}")
    perturbed = normalize(field + args.eps * mode)
    apply_boundary_(perturbed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, perturbed.detach().cpu().numpy().astype(np.float32))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
