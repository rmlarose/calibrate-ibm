#!/usr/bin/env python3
"""Run one step of a warm-start chain on Stanford (round 2).

Two chains:
  - 12_be2_f20 (92q): continue cumul_2_4 from ICER checkpoint, then cumul_2_4_5
  - f18_dd (114q): new chain on DD+postselectprep data

Usage:
    python run_chain_stanford2.py FRAGMENT STEP_INDEX
    python run_chain_stanford2.py --dry-run
"""
import os
import sys
import pickle
import numpy as np

BASE = "/home/rowlan91/ben/calibrate-ibm/experiments"
SQD_SCRIPT = os.path.join(BASE, "ATP/sqd_sbd/run_sqd_sbd.py")
SBD_EXE = "/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
PYTHON = "/home/rowlan91/miniforge3/envs/sqd/bin/python"
RESULTS_BASE = "/home/rowlan91/ben/warm_start_experiment/results_stanford2"

FRAGMENTS = {
    '12_be2_f20': {
        'fragment': 'atp_12_be2_f20',
        'circuit_dir': os.path.join(BASE, "ATP/circuits/its_0_and_12"),
        'hamiltonian_dir': os.path.join(BASE, "ATP/hamiltonians"),
        'results_dir': os.path.join(BASE, "ATP/results/its_0_and_12"),
        'output_base': os.path.join(RESULTS_BASE, "12_be2_f20/hardware"),
        'num_gpus': 2,
        'chain': [
            # cumul_2_4 resumes from ICER checkpoint (already in place)
            ('cumulative_2_4',       [2, 4],        None),
            ('cumulative_2_4_5',     [2, 4, 5],     'cumulative_2_4'),
        ],
    },
    'f18_dd': {
        'fragment': 'atp_0_be2_f18',
        'circuit_dir': os.path.join(BASE, "ATP/circuits/truncated_pool"),
        'hamiltonian_dir': os.path.join(BASE, "ATP/hamiltonians"),
        'results_dir': os.path.join(BASE, "ATP/results/truncated_pool/atp_0_be2_f18_dd_boston"),
        'output_base': os.path.join(RESULTS_BASE, "f18_dd/hardware"),
        'num_gpus': 2,
        'chain': [
            ('singleton_1',          [1],              None),
            ('cumulative_1_2',       [1, 2],           'singleton_1'),
            ('cumulative_1_2_3',     [1, 2, 3],        'cumulative_1_2'),
            ('cumulative_1_2_3_5',   [1, 2, 3, 5],     'cumulative_1_2_3'),
        ],
    },
}


def iters_key(adapts):
    return "_".join(str(a) for a in adapts)


def prepare_warm_checkpoint(source_path, dest_path):
    """Copy checkpoint preserving warm state, reset iteration tracking."""
    with open(source_path, "rb") as f:
        ckpt = pickle.load(f)
    print("  Source: %s" % source_path)
    print("  next_iteration: %d, carryover: %d strings" % (
        ckpt.get("next_iteration", 0),
        len(ckpt["carryover_a"]) if ckpt.get("carryover_a") is not None else 0))
    ckpt["next_iteration"] = 0
    ckpt["iteration_energies"] = []
    ckpt["all_occupancies"] = []
    ckpt["iteration_stats"] = []
    ckpt["rng"] = np.random.default_rng(42)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        pickle.dump(ckpt, f)
    print("  Warm checkpoint written: %s" % dest_path)


def find_checkpoint(output_base, name, adapts):
    key = iters_key(adapts)
    return os.path.join(output_base, name, "checkpoint_%s.pkl" % key)


def main():
    if "--dry-run" in sys.argv:
        for frag_name, cfg in FRAGMENTS.items():
            print("=== %s (%d GPUs) ===" % (frag_name, cfg['num_gpus']))
            for i, (name, adapts, seed_from) in enumerate(cfg['chain']):
                if seed_from is None:
                    src = "(cold start / resume existing)"
                else:
                    prev_idx = [c[0] for c in cfg['chain']].index(seed_from)
                    prev_adapts = cfg['chain'][prev_idx][1]
                    src = find_checkpoint(cfg['output_base'], seed_from, prev_adapts)
                print("  [%d] %s (adapts %s) <- %s" % (i, name, adapts, src))
            print()
        return

    frag_name = sys.argv[1]
    step_idx = int(sys.argv[2])

    cfg = FRAGMENTS[frag_name]
    chain = cfg['chain']
    num_gpus = cfg['num_gpus']

    if step_idx < 0 or step_idx >= len(chain):
        print("ERROR: step %d out of range [0, %d]" % (step_idx, len(chain) - 1))
        sys.exit(1)

    name, adapts, seed_from = chain[step_idx]
    key = iters_key(adapts)
    output_dir = os.path.join(cfg['output_base'], name)
    dest_ckpt = os.path.join(output_dir, "checkpoint_%s.pkl" % key)

    print("=" * 60)
    print("Fragment: %s, Step %d: %s" % (frag_name, step_idx, name))
    print("ADAPT iterations: %s" % adapts)
    print("NUM_GPUS: %d" % num_gpus)
    print("Output: %s" % output_dir)
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    if seed_from is not None:
        if os.path.isfile(dest_ckpt):
            with open(dest_ckpt, "rb") as f:
                existing = pickle.load(f)
            existing_iters = existing.get("next_iteration", 0)
            if existing_iters > 0:
                print("\nResuming in-progress checkpoint: %s (iteration %d)" % (
                    dest_ckpt, existing_iters))
            else:
                print("\nWarm-start checkpoint exists but has 0 iterations, re-preparing...")
                prev_idx = [c[0] for c in chain].index(seed_from)
                prev_adapts = chain[prev_idx][1]
                src_ckpt = find_checkpoint(cfg['output_base'], seed_from, prev_adapts)
                if not os.path.isfile(src_ckpt):
                    print("ERROR: source checkpoint not found: %s" % src_ckpt)
                    sys.exit(1)
                prepare_warm_checkpoint(src_ckpt, dest_ckpt)
        else:
            prev_idx = [c[0] for c in chain].index(seed_from)
            prev_adapts = chain[prev_idx][1]
            src_ckpt = find_checkpoint(cfg['output_base'], seed_from, prev_adapts)
            if not os.path.isfile(src_ckpt):
                print("ERROR: source checkpoint not found: %s" % src_ckpt)
                sys.exit(1)
            print("\nPreparing warm-start checkpoint...")
            prepare_warm_checkpoint(src_ckpt, dest_ckpt)

    cmd = [
        PYTHON, SQD_SCRIPT,
        "--fragment", cfg['fragment'],
        "--circuit_dir", cfg['circuit_dir'],
        "--hamiltonian_dir", cfg['hamiltonian_dir'],
        "--results_dir", cfg['results_dir'],
        "--sbd_exe", SBD_EXE,
        "--output_dir", output_dir,
        "--max_iterations", "5000",
        "--energy_tol", "1e-6",
        "--occupancies_tol", "1e-6",
        "--num_gpus", str(num_gpus),
        "--resume",
    ] + [str(a) for a in adapts]

    print("\nCommand: %s" % " ".join(cmd))
    sys.stdout.flush()

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
