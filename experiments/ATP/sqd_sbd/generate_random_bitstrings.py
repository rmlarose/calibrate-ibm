"""Generate IID random bitstrings for SQD control experiments.

Two modes:
  --mode hamming : each half has exactly nalpha/nbeta ones (correct Hamming weight)
  --mode iid     : each bit independently 0 or 1 (truly random)

Bitstrings use Qiskit convention: <beta_reversed><alpha_reversed>, 64 bits total.
"""
import argparse
import pickle
import numpy as np
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--mode", choices=["hamming", "iid"], required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--norb", type=int, default=32)
    parser.add_argument("--nalpha", type=int, default=16)
    parser.add_argument("--nbeta", type=int, default=16)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"counts_random_{args.mode}_{args.num_samples}.pkl"

    rng = np.random.default_rng(args.seed)
    nqubits = 2 * args.norb

    print(f"Generating {args.num_samples} random bitstrings (mode={args.mode}, seed={args.seed})")

    bitstrings = []
    if args.mode == "iid":
        for _ in range(args.num_samples):
            bits = rng.integers(0, 2, size=nqubits)
            bitstrings.append("".join(str(b) for b in bits))
    else:
        # Correct Hamming weight: shuffle fixed-weight templates
        alpha_base = np.array([1]*args.nalpha + [0]*(args.norb - args.nalpha))
        beta_base = np.array([1]*args.nbeta + [0]*(args.norb - args.nbeta))
        for _ in range(args.num_samples):
            rng.shuffle(alpha_base)
            rng.shuffle(beta_base)
            bs = "".join(str(b) for b in beta_base) + "".join(str(b) for b in alpha_base)
            bitstrings.append(bs)

    counts = dict(Counter(bitstrings))

    print(f"  {len(counts)} unique bitstrings from {args.num_samples} samples")
    print(f"  Top 5:")
    for bs, cnt in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {bs}: {cnt}")

    with open(args.output, "wb") as f:
        pickle.dump(counts, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
