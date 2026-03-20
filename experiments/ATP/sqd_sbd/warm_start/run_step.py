#!/usr/bin/env python3
"""Run one step of the warm-start serial chain.

Each step warm-starts from the previous step's converged checkpoint
(or from the seed singleton for step 0).

Usage:
    python run_step.py STEP_INDEX
    python run_step.py --dry-run          # print all 5 steps
"""
import os
import sys
import subprocess

BASE = "/home/rowlan91/ben/warm_start_experiment"
SQD_SCRIPT = os.path.join(BASE, "run_sqd_sbd.py")
SBD_EXE = "/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
PYTHON = "/home/rowlan91/miniforge3/envs/sqd/bin/python"
PREPARE_SCRIPT = os.path.join(BASE, "prepare_warm_checkpoint.py")

INPUT = os.path.join(BASE, "input")
RESULTS = os.path.join(BASE, "results")
SEED_DIR = os.path.join(BASE, "seed")

NUM_GPUS = 4

# ── Chain definition ──────────────────────────────────────────────────────────
# Each step: (name, adapt_iters, seed_source)
# seed_source is either "seed" (singleton checkpoint) or previous step name.
CHAIN = [
    ("cumul_1_2",           [1, 2],              "seed"),
    ("cumul_1_2_3",         [1, 2, 3],           "cumul_1_2"),
    ("cumul_1_2_3_4",       [1, 2, 3, 4],        "cumul_1_2_3"),
    ("cumul_1_2_3_4_5",     [1, 2, 3, 4, 5],     "cumul_1_2_3_4"),
    ("cumul_1_2_3_4_5_10",  [1, 2, 3, 4, 5, 10], "cumul_1_2_3_4_5"),
]


def iters_key(adapts):
    return "_".join(str(a) for a in adapts)


def find_source_checkpoint(step_idx):
    """Find the source checkpoint for a given step."""
    name, adapts, seed_from = CHAIN[step_idx]

    if seed_from == "seed":
        # Seed from singleton_1 checkpoint
        return os.path.join(SEED_DIR, "checkpoint_1.pkl")
    else:
        # Seed from previous step's converged checkpoint
        prev_idx = [c[0] for c in CHAIN].index(seed_from)
        prev_name, prev_adapts, _ = CHAIN[prev_idx]
        prev_key = iters_key(prev_adapts)
        return os.path.join(RESULTS, prev_name, "checkpoint_%s.pkl" % prev_key)


def build_command(step_idx):
    """Build the run_sqd_sbd.py command for a chain step."""
    name, adapts, _ = CHAIN[step_idx]
    key = iters_key(adapts)
    output_dir = os.path.join(RESULTS, name)

    cmd = [
        PYTHON, SQD_SCRIPT,
        "--fragment", "atp_0_be2_f2",
        "--circuit_dir", os.path.join(INPUT, "circuits", "truncated_pool"),
        "--hamiltonian_dir", os.path.join(INPUT, "hamiltonians"),
        "--results_dir", os.path.join(INPUT, "results", "truncated_pool"),
        "--sbd_exe", SBD_EXE,
        "--output_dir", output_dir,
        "--max_iterations", "5000",
        "--num_gpus", str(NUM_GPUS),
        "--resume",
    ] + [str(a) for a in adapts]
    return cmd


def main():
    if "--dry-run" in sys.argv:
        for i in range(len(CHAIN)):
            name, adapts, seed_from = CHAIN[i]
            src = find_source_checkpoint(i)
            key = iters_key(adapts)
            dest = os.path.join(RESULTS, name, "checkpoint_%s.pkl" % key)
            cmd = build_command(i)
            print("[Step %d] %s" % (i, name))
            print("  Seed: %s" % src)
            print("  Dest: %s" % dest)
            print("  Cmd:  %s" % " ".join(cmd))
            print()
        return

    step_idx = int(sys.argv[1])
    if step_idx < 0 or step_idx >= len(CHAIN):
        print("ERROR: step_idx %d out of range [0, %d]" % (step_idx, len(CHAIN) - 1))
        sys.exit(1)

    name, adapts, _ = CHAIN[step_idx]
    key = iters_key(adapts)
    output_dir = os.path.join(RESULTS, name)
    dest_ckpt = os.path.join(output_dir, "checkpoint_%s.pkl" % key)

    # Step 1: Find and verify source checkpoint
    src_ckpt = find_source_checkpoint(step_idx)
    if not os.path.isfile(src_ckpt):
        print("ERROR: source checkpoint not found: %s" % src_ckpt)
        print("The previous chain step may not have converged yet.")
        sys.exit(1)

    # Step 2: Prepare warm-start checkpoint
    print("=" * 60)
    print("Chain step %d: %s" % (step_idx, name))
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    print("\nPreparing warm-start checkpoint...")
    prep_cmd = [
        PYTHON, PREPARE_SCRIPT,
        "--source", src_ckpt,
        "--dest", dest_ckpt,
    ]
    subprocess.check_call(prep_cmd)

    # Step 3: Run SQD
    print("\nStarting SQD run...")
    cmd = build_command(step_idx)
    print("Command: %s" % " ".join(cmd))
    print(flush=True)

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
