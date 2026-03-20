#!/usr/bin/env python3
"""Prepare a warm-start checkpoint from a converged run's checkpoint.

Copies the converged carryover determinants and occupancies but resets
iteration tracking so the warm-started run has clean bookkeeping.

Usage:
    python prepare_warm_checkpoint.py --source SRC.pkl --dest DEST.pkl
"""
import argparse
import os
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Path to converged checkpoint to seed from")
    parser.add_argument("--dest", required=True,
                        help="Path for the warm-start checkpoint")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print("ERROR: source checkpoint not found: %s" % args.source)
        raise SystemExit(1)

    with open(args.source, "rb") as f:
        ckpt = pickle.load(f)

    print("Source checkpoint: %s" % args.source)
    print("  next_iteration: %d" % ckpt.get("next_iteration", 0))
    print("  best_ever_energy: %s" % ckpt.get("best_ever_energy"))
    co_a = ckpt.get("carryover_a")
    co_b = ckpt.get("carryover_b")
    print("  carryover_a size: %d" % (len(co_a) if co_a is not None else 0))
    print("  carryover_b size: %d" % (len(co_b) if co_b is not None else 0))

    # Reset iteration tracking (clean bookkeeping for the new run)
    ckpt["next_iteration"] = 0
    ckpt["iteration_energies"] = []
    ckpt["all_occupancies"] = []
    ckpt["iteration_stats"] = []
    # Keep best_ever_energy from source (run_sqd_sbd.py expects a float for resume print)

    # Fresh RNG for independence from source run
    ckpt["rng"] = np.random.default_rng(42)

    # Write to destination
    os.makedirs(os.path.dirname(os.path.abspath(args.dest)), exist_ok=True)
    with open(args.dest, "wb") as f:
        pickle.dump(ckpt, f)

    print("\nWarm-start checkpoint written: %s" % args.dest)
    print("  Preserved: carryover_a/b, current_occupancies, current_energy, current_occ")
    print("  Reset: next_iteration=0, iteration_energies=[], best_ever_energy=None, rng=fresh")


if __name__ == "__main__":
    main()
