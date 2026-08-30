"""
Compute Inverse Participation Ratio (IPR) for converged SQD runs.

For each converged checkpoint, reproduces one SQD iteration (using the saved
occupancies and RNG state) and captures the CI wavefunction amplitudes from
SBD's --dump_matrix_form_wf output.  Then computes:

    IPR = 1 / sum_i |c_i|^4

where c_i are the CI coefficients in the determinant basis.
IPR ~ effective number of determinants contributing to the wavefunction.

Usage:
    python compute_ipr.py --sbd_exe <path> --checkpoint <ckpt.pkl> \
        --semiclassical_counts <counts.pkl> [adapt_iterations ...]
"""

import argparse
import collections
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile

import numpy as np
import pyscf.tools

from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample


# Reuse helpers from run_sqd_semiclassical.py
def ci_strings_to_file(ci_strs, norb, filepath):
    with open(filepath, 'w') as f:
        for val in ci_strs:
            bits = format(int(val), f'0{norb}b')
            f.write(bits + '\n')


def parse_sbd_output(stdout):
    energy = None
    for line in stdout.split('\n'):
        if 'Sample-based diagonalization: Energy =' in line:
            energy = float(line.split('=')[1].strip())
    return energy


def parse_wavefunction(wf_path):
    """Parse SBD's wf.txt and return dict of {(ia, ib): amplitude}."""
    amplitudes = {}
    with open(wf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            amp_str, rest = line.split(" # ", 1)
            amp = float(amp_str)
            parts = rest.split()
            ia = int(parts[0].rstrip(':'))
            ib = int(parts[2].rstrip(':'))
            amplitudes[(ia, ib)] = amp
    return amplitudes


def compute_ipr(amplitudes):
    """Compute IPR = 1 / sum(|c_i|^4) from amplitude dict."""
    coeffs = np.array(list(amplitudes.values()))
    # Normalize (should already be normalized, but just in case)
    norm = np.sqrt(np.sum(coeffs**2))
    if norm > 0:
        coeffs = coeffs / norm
    sum_c4 = np.sum(coeffs**4)
    if sum_c4 == 0:
        return 0.0
    return 1.0 / sum_c4


def transform_bitstring(bits):
    left = [bits[i] for i in range(len(bits)) if i % 2 == 1]
    right = [bits[i] for i in range(len(bits)) if i % 2 == 0]
    left.reverse()
    right.reverse()
    return ''.join(left + right)


def load_and_process_bitstrings(fragment, circuit_dir, hamiltonian_dir, results_dir,
                                 adapt_iterations, n_orbitals, num_electrons):
    nqubits = 2 * n_orbitals
    all_counts = []
    for adapt_iter in adapt_iterations:
        fname = glob.glob(f"{results_dir}/{fragment}/*{adapt_iter:03d}*.qasm*")[0]
        counts = pickle.load(open(fname, "rb"))
        mode_order = pickle.load(
            open(f"{circuit_dir}/{fragment}/{fragment}_mode_order_{adapt_iter:03d}_adaptiterations.pkl", "rb")
        )
        qubit_order = pickle.load(
            open(f"{circuit_dir}/{fragment}/{fragment}_qubit_order_{adapt_iter:03d}_adaptiterations.pkl", "rb")
        )
        permuted = {}
        for orig_bs in counts.keys():
            qp = "".join([orig_bs[qubit_order.index(n)] for n in range(nqubits)])
            mp = "".join([qp[mode_order.index(n)] for n in range(nqubits)])
            final = transform_bitstring(mp)
            permuted[final] = counts[orig_bs]
        all_counts.append(permuted)
    merged = collections.Counter()
    for c in all_counts:
        for bs, count in c.items():
            merged[bs] += count
    return merged


def bitstrings_to_matrix(counts, nqubits):
    total = sum(counts.values())
    bitstrings = []
    probs = []
    for bs, count in counts.items():
        row = np.array([int(b) for b in bs], dtype=np.bool_)
        bitstrings.append(row)
        probs.append(count / total)
    return np.array(bitstrings), np.array(probs)


def bitstring_matrix_to_integers(matrix):
    n = matrix.shape[1]
    powers = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    return matrix.astype(np.int64) @ powers


def extract_ci_strings(batch, n_orbitals):
    alpha_part = batch[:, n_orbitals:]
    beta_part = batch[:, :n_orbitals]
    alpha_ints = set(bitstring_matrix_to_integers(alpha_part).tolist())
    beta_ints = set(bitstring_matrix_to_integers(beta_part).tolist())
    all_strings = alpha_ints | beta_ints
    return np.array(sorted(all_strings), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Compute IPR for a converged SQD checkpoint")
    parser.add_argument("--fragment", type=str, default="atp_0_be2_f4")
    parser.add_argument("--circuit_dir", type=str, default="../../circuits")
    parser.add_argument("--hamiltonian_dir", type=str, default="../../hamiltonians")
    parser.add_argument("--results_dir", type=str, default="../../results")
    parser.add_argument("--sbd_exe", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint pickle")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for IPR results")
    parser.add_argument("--samples_per_batch", type=int, default=500)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--carryover_threshold", type=float, default=1e-5)
    parser.add_argument("--semiclassical_counts", type=str, default=None)
    parser.add_argument("adapt_iterations", type=int, nargs='*', default=[])
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)

    converged_energy = ckpt['current_energy']
    current_occupancies = ckpt['current_occupancies']
    carryover_a = ckpt['carryover_a']
    carryover_b = ckpt['carryover_b']
    rng = ckpt['rng']
    n_iters = ckpt['next_iteration']
    print(f"  Converged energy: {converged_energy:.10f} Ha after {n_iters} iterations")
    print(f"  Carryover: {len(carryover_a)} alpha, {len(carryover_b)} beta strings")

    # Read Hamiltonian
    fcidump_path = f"{args.hamiltonian_dir}/{args.fragment}.fcidump"
    fcidump = pyscf.tools.fcidump.read(fcidump_path)
    n_orbitals = fcidump.get("NORB")
    num_electrons = fcidump.get("NELEC")
    ecore = fcidump.get("ECORE")
    n_alpha = num_electrons // 2
    n_beta = num_electrons // 2

    # Load bitstrings (same as the original run)
    merged_counts = collections.Counter()
    if args.semiclassical_counts:
        with open(args.semiclassical_counts, 'rb') as f:
            sc_counts = pickle.load(f)
        for bs, count in sc_counts.items():
            merged_counts[bs] += count

    if args.adapt_iterations:
        hw_counts = load_and_process_bitstrings(
            args.fragment, args.circuit_dir, args.hamiltonian_dir,
            args.results_dir, args.adapt_iterations, n_orbitals, num_electrons
        )
        for bs, count in hw_counts.items():
            merged_counts[bs] += count

    raw_bs_matrix, raw_probs = bitstrings_to_matrix(merged_counts, 2 * n_orbitals)

    # Reproduce one iteration with saved occupancies + RNG
    print("\nRunning config recovery...")
    bitstrings, probs = recover_configurations(
        raw_bs_matrix, raw_probs, current_occupancies,
        n_alpha, n_beta, rand_seed=rng
    )
    print(f"  After recovery: {len(bitstrings)} unique bitstrings")

    batches = subsample(
        bitstrings, probs,
        samples_per_batch=args.samples_per_batch,
        num_batches=args.num_batches,
        rand_seed=rng,
    )

    # Diagonalize each batch, capturing wavefunction
    results = {"checkpoint": args.checkpoint, "converged_energy": converged_energy,
               "n_iterations": n_iters, "batches": []}

    for b, batch in enumerate(batches):
        ci_strings = extract_ci_strings(batch, n_orbitals)

        # Merge with carryover
        all_a = ci_strings
        if carryover_a is not None and len(carryover_a) > 0:
            all_a = np.unique(np.concatenate([carryover_a, ci_strings]))

        n_ci = len(all_a)
        n_dets = n_ci ** 2
        print(f"\n  Batch {b}: {n_ci} unique CI strings ({n_dets} determinants)")

        with tempfile.TemporaryDirectory() as tmpdir:
            adet_path = os.path.join(tmpdir, "AlphaDets.txt")
            ci_strings_to_file(all_a, n_orbitals, adet_path)
            wf_path = os.path.join(tmpdir, "wf.txt")

            mpirun = os.environ.get("MPIRUN", "mpirun")
            cmd = [
                mpirun, "-np", "1",
                args.sbd_exe,
                "--fcidump", fcidump_path,
                "--adetfile", adet_path,
                "--method", "0",
                "--block", "10",
                "--iteration", "10000",
                "--tolerance", "1e-9",
                "--rdm", "0",
                "--use_precalculated_dets", "1",
                "--dump_matrix_form_wf", wf_path,
            ]

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                print(f"  SBD STDERR: {result.stderr}", file=sys.stderr)
                raise RuntimeError(f"SBD failed for batch {b}")

            energy = parse_sbd_output(result.stdout)
            total_energy = energy + ecore
            print(f"  Batch {b}: energy = {total_energy:.10f} Ha")

            # Parse wavefunction and compute IPR
            amplitudes = parse_wavefunction(wf_path)
            ipr = compute_ipr(amplitudes)
            n_amps = len(amplitudes)

            # Also compute participation at various thresholds
            coeffs = np.array(list(amplitudes.values()))
            norm = np.sqrt(np.sum(coeffs**2))
            if norm > 0:
                coeffs = coeffs / norm

            n_above_1e2 = int(np.sum(coeffs**2 > 1e-2))
            n_above_1e3 = int(np.sum(coeffs**2 > 1e-3))
            n_above_1e4 = int(np.sum(coeffs**2 > 1e-4))
            n_above_1e5 = int(np.sum(coeffs**2 > 1e-5))

            print(f"  Batch {b}: IPR = {ipr:.1f}, n_amps = {n_amps}, n_dets = {n_dets}")
            print(f"    |c|^2 > 1e-2: {n_above_1e2}")
            print(f"    |c|^2 > 1e-3: {n_above_1e3}")
            print(f"    |c|^2 > 1e-4: {n_above_1e4}")
            print(f"    |c|^2 > 1e-5: {n_above_1e5}")

            results["batches"].append({
                "batch": b,
                "energy": total_energy,
                "n_ci_strings": n_ci,
                "n_determinants": n_dets,
                "n_amplitudes": n_amps,
                "ipr": ipr,
                "n_above_1e2": n_above_1e2,
                "n_above_1e3": n_above_1e3,
                "n_above_1e4": n_above_1e4,
                "n_above_1e5": n_above_1e5,
            })

    # Average IPR across batches
    avg_ipr = np.mean([b["ipr"] for b in results["batches"]])
    results["avg_ipr"] = avg_ipr
    print(f"\n  Average IPR: {avg_ipr:.1f}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved IPR results to {args.output}")


if __name__ == "__main__":
    main()
