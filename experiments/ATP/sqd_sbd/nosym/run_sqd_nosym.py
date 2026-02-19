"""
SQD loop using SBD GPU solver — NO spin symmetrization.

This is a copy of run_sqd_sbd.py with one key difference:
instead of pooling alpha and beta CI strings into a single list
(symmetrize_spin=True), we keep them separate and pass them to SBD
as --adetfile and --bdetfile independently.

This means the CI space is |unique_alpha| x |unique_beta| rather than
|union(alpha, beta)|^2. The space is smaller and does not include
cross-spin combinations that were never measured.

Dependencies: numpy, pyscf, qiskit-addon-sqd, matplotlib
"""

import argparse
import subprocess
import tempfile
import os
import re
import pickle
import collections
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyscf.tools
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample


# ---------- SBD interface ----------

def ci_strings_to_file(ci_strs, norb, filepath):
    """Write CI strings as binary text for SBD."""
    with open(filepath, 'w') as f:
        for val in ci_strs:
            f.write(format(int(val), f'0{norb}b') + '\n')


def parse_sbd_output(stdout):
    """Parse energy and spin-resolved density from SBD stdout."""
    energy = density_alpha = density_beta = None
    for line in stdout.split('\n'):
        if 'Energy =' in line:
            energy = float(line.split('=')[1].strip())
        elif 'density_alpha =' in line:
            m = re.search(r'\[(.+)\]', line)
            if m:
                density_alpha = np.array([float(x) for x in m.group(1).split(',')])
        elif 'density_beta =' in line:
            m = re.search(r'\[(.+)\]', line)
            if m:
                density_beta = np.array([float(x) for x in m.group(1).split(',')])
    return energy, density_alpha, density_beta


def extract_carryover(wf_path, adet_path, bdet_path, threshold):
    """Extract carryover determinants from SBD wavefunction dump.

    Unlike the symmetrized version, alpha and beta det files are separate,
    so we read each independently.
    """
    with open(adet_path) as f:
        adet_strs = [line.strip() for line in f if line.strip()]
    with open(bdet_path) as f:
        bdet_strs = [line.strip() for line in f if line.strip()]

    alpha_idx, beta_idx = set(), set()
    with open(wf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            amp_str, rest = line.split(" # ", 1)
            if abs(float(amp_str)) > threshold:
                parts = rest.split()
                alpha_idx.add(int(parts[0].rstrip(':')))
                beta_idx.add(int(parts[2].rstrip(':')))

    co_a = np.array([int(adet_strs[i], 2) for i in sorted(alpha_idx)], dtype=np.int64)
    co_b = np.array([int(bdet_strs[i], 2) for i in sorted(beta_idx)], dtype=np.int64)
    return co_a, co_b


def run_sbd(ci_alpha, ci_beta, fcidump_path, sbd_exe, norb,
            carryover_alpha=None, carryover_beta=None,
            carryover_threshold=1e-5,
            block=10, tolerance=1e-9, max_davidson=10000):
    """Run SBD with separate alpha/beta determinant lists.

    Unlike the symmetrized version, we pass --adetfile and --bdetfile
    separately to SBD.

    Returns (energy, occ_alpha, occ_beta, carryover_a, carryover_b).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        adet_path = os.path.join(tmpdir, "AlphaDets.txt")
        bdet_path = os.path.join(tmpdir, "BetaDets.txt")
        wf_path = os.path.join(tmpdir, "wf.txt")

        # Merge carryover into alpha/beta separately
        all_alpha = ci_alpha
        if carryover_alpha is not None and len(carryover_alpha) > 0:
            all_alpha = np.unique(np.concatenate([carryover_alpha, ci_alpha]))
        ci_strings_to_file(all_alpha, norb, adet_path)

        all_beta = ci_beta
        if carryover_beta is not None and len(carryover_beta) > 0:
            all_beta = np.unique(np.concatenate([carryover_beta, ci_beta]))
        ci_strings_to_file(all_beta, norb, bdet_path)

        cmd = [
            os.environ.get("MPIRUN", "mpirun"), "-np", "1",
            sbd_exe,
            "--fcidump", fcidump_path,
            "--adetfile", adet_path,
            "--bdetfile", bdet_path,
            "--method", "0",
            "--block", str(block),
            "--iteration", str(max_davidson),
            "--tolerance", str(tolerance),
            "--rdm", "0",
            "--use_precalculated_dets", "1",
            "--dump_matrix_form_wf", wf_path,
        ]

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print(f"SBD STDERR: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"SBD failed (rc={result.returncode})")

        energy, occ_a, occ_b = parse_sbd_output(result.stdout)
        if energy is None:
            raise RuntimeError("Failed to parse energy from SBD output")

        co_a = co_b = np.array([], dtype=np.int64)
        if os.path.exists(wf_path):
            co_a, co_b = extract_carryover(wf_path, adet_path, bdet_path, carryover_threshold)

        return energy, occ_a, occ_b, co_a, co_b


# ---------- Bitstring processing ----------

def transform_bitstring(bits):
    """Openfermion convention -> Qiskit convention."""
    left = [bits[i] for i in range(len(bits)) if i % 2 == 1]
    right = [bits[i] for i in range(len(bits)) if i % 2 == 0]
    left.reverse()
    right.reverse()
    return ''.join(left + right)


def load_bitstrings(fragment, circuit_dir, results_dir, adapt_iterations, nqubits):
    """Load and permute measurement bitstrings from all ADAPT iterations."""
    merged = collections.Counter()
    for adapt_iter in adapt_iterations:
        fname = glob.glob(f"{results_dir}/{fragment}/*{adapt_iter:03d}*.qasm*")[0]
        counts = pickle.load(open(fname, "rb"))
        mode_order = pickle.load(open(
            f"{circuit_dir}/{fragment}/{fragment}_mode_order_{adapt_iter:03d}_adaptiterations.pkl", "rb"))
        qubit_order = pickle.load(open(
            f"{circuit_dir}/{fragment}/{fragment}_qubit_order_{adapt_iter:03d}_adaptiterations.pkl", "rb"))

        for bs in counts:
            qp = "".join([bs[qubit_order.index(n)] for n in range(nqubits)])
            mp = "".join([qp[mode_order.index(n)] for n in range(nqubits)])
            merged[transform_bitstring(mp)] += counts[bs]

        print(f"  ADAPT iter {adapt_iter}: {len(counts)} unique, {sum(counts.values())} shots")
    return merged


def counts_to_matrix(counts, nqubits):
    """Convert counts dict to (bitstring_matrix, probabilities)."""
    total = sum(counts.values())
    bs_list = list(counts.keys())
    matrix = np.array([[int(b) for b in bs] for bs in bs_list], dtype=np.bool_)
    probs = np.array([counts[bs] / total for bs in bs_list])
    return matrix, probs


def extract_ci_strings_nosym(batch, n_orbitals):
    """Extract SEPARATE alpha and beta CI strings (no symmetrization).

    Unlike extract_ci_strings in run_sqd_sbd.py which takes the union of alpha
    and beta halves, this returns them separately. The CI space is then
    |unique_alpha| x |unique_beta| instead of |union|^2.
    """
    powers = 1 << np.arange(n_orbitals - 1, -1, -1, dtype=np.int64)
    alpha = batch[:, n_orbitals:].astype(np.int64) @ powers
    beta = batch[:, :n_orbitals].astype(np.int64) @ powers
    ci_alpha = np.array(sorted(set(alpha.tolist())), dtype=np.int64)
    ci_beta = np.array(sorted(set(beta.tolist())), dtype=np.int64)
    return ci_alpha, ci_beta


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="SQD loop with SBD GPU solver (no spin symmetrization)")
    p.add_argument("--fragment", default="atp_0_be2_f4")
    p.add_argument("--circuit_dir", default="../../circuits")
    p.add_argument("--hamiltonian_dir", default="../../hamiltonians")
    p.add_argument("--results_dir", default="../../results")
    p.add_argument("--sbd_exe", required=True,
                   help="Path to SBD diag executable")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--samples_per_batch", type=int, default=500)
    p.add_argument("--num_batches", type=int, default=2)
    p.add_argument("--max_iterations", type=int, default=100)
    p.add_argument("--energy_tol", type=float, default=1e-8)
    p.add_argument("--occupancies_tol", type=float, default=1e-8)
    p.add_argument("--carryover_threshold", type=float, default=1e-5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("adapt_iterations", type=int, nargs='+')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    iters_key = "_".join(map(str, args.adapt_iterations))
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{iters_key}.pkl")
    energy_file = os.path.join(args.output_dir, f"sqd_energies_{iters_key}.txt")

    # Read Hamiltonian
    fcidump_path = f"{args.hamiltonian_dir}/{args.fragment}.fcidump"
    fcidump = pyscf.tools.fcidump.read(fcidump_path)
    n_orbitals = fcidump.get("NORB")
    num_electrons = fcidump.get("NELEC")
    ecore = fcidump.get("ECORE")
    n_alpha = num_electrons // 2
    n_beta = num_electrons // 2
    print(f"NORB={n_orbitals}, NELEC={num_electrons}, ECORE={ecore}")

    # Load measurement data
    print("Loading measurement data...")
    counts = load_bitstrings(
        args.fragment, args.circuit_dir, args.results_dir,
        args.adapt_iterations, 2 * n_orbitals)
    print(f"  Total: {len(counts)} unique bitstrings, {sum(counts.values())} shots")

    raw_bs, raw_probs = counts_to_matrix(counts, 2 * n_orbitals)

    # Initialize state
    rng = np.random.default_rng(42)
    occupancies = None
    carryover_alpha = None
    carryover_beta = None
    best_energy = None
    prev_energy = None
    prev_occ = None
    converged = False
    energies = []
    start = 0

    # Resume
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = pickle.load(open(checkpoint_path, 'rb'))
        energies = ckpt['energies']
        carryover_alpha = ckpt['carryover_alpha']
        carryover_beta = ckpt['carryover_beta']
        best_energy = ckpt['best_energy']
        prev_energy = ckpt['prev_energy']
        prev_occ = ckpt['prev_occ']
        occupancies = ckpt['occupancies']
        rng = ckpt['rng']
        start = ckpt['next_iter']
        print(f"Resuming at iteration {start + 1}, best energy: {best_energy:.10f} Ha")

    print(f"\nSQD loop (NO symmetrization): {args.num_batches} batches x "
          f"{args.samples_per_batch} samples, max {args.max_iterations} iterations")
    sys.stdout.flush()

    for it in range(start, args.max_iterations):
        print(f"\n--- Iteration {it + 1} ---")
        sys.stdout.flush()

        # Config recovery or postselection
        if occupancies is not None:
            bs, pr = recover_configurations(
                raw_bs, raw_probs, occupancies, n_alpha, n_beta, rand_seed=rng)
        else:
            bs, pr = postselect_by_hamming_right_and_left(
                raw_bs, raw_probs, hamming_right=n_alpha, hamming_left=n_beta)
        print(f"  Pool: {len(bs)} bitstrings")

        # Subsample
        batches = subsample(bs, pr, args.samples_per_batch, args.num_batches, rand_seed=rng)

        # Diagonalize each batch
        batch_results = []
        for b, batch in enumerate(batches):
            ci_alpha, ci_beta = extract_ci_strings_nosym(batch, n_orbitals)
            print(f"  Batch {b}: {len(ci_alpha)} alpha x {len(ci_beta)} beta "
                  f"= {len(ci_alpha)*len(ci_beta)} dets")
            sys.stdout.flush()

            e, oa, ob, co_a, co_b = run_sbd(
                ci_alpha, ci_beta, fcidump_path, args.sbd_exe, n_orbitals,
                carryover_alpha=carryover_alpha, carryover_beta=carryover_beta,
                carryover_threshold=args.carryover_threshold)

            total_e = e + ecore
            print(f"  Batch {b}: energy = {total_e:.10f}")
            batch_results.append((total_e, oa, ob, co_a, co_b))

        # Best batch
        best_idx = min(range(len(batch_results)), key=lambda i: batch_results[i][0])
        e, oa, ob, co_a, co_b = batch_results[best_idx]
        energies.append(e)
        print(f"  Best: {e:.10f} Ha")

        # Update carryover (separate alpha/beta — NOT symmetrized)
        if len(co_a) > 0:
            if carryover_alpha is not None:
                carryover_alpha = np.unique(np.concatenate([carryover_alpha, co_a]))
            else:
                carryover_alpha = co_a
        if len(co_b) > 0:
            if carryover_beta is not None:
                carryover_beta = np.unique(np.concatenate([carryover_beta, co_b]))
            else:
                carryover_beta = co_b
        n_co_a = len(carryover_alpha) if carryover_alpha is not None else 0
        n_co_b = len(carryover_beta) if carryover_beta is not None else 0
        print(f"  Carryover: {n_co_a} alpha, {n_co_b} beta")

        if best_energy is None or e < best_energy:
            best_energy = e

        occupancies = (oa, ob)

        # Convergence check
        converged = False
        if prev_energy is not None:
            de = abs(prev_energy - e)
            do = np.linalg.norm(
                np.concatenate(prev_occ) - np.concatenate([oa, ob]), ord=np.inf)
            print(f"  dE={de:.2e}, dOcc={do:.2e}")
            if de < args.energy_tol and do < args.occupancies_tol:
                print(f"\n  CONVERGED at iteration {it + 1}!")
                converged = True

        prev_energy = e
        prev_occ = (oa, ob)

        # Checkpoint
        np.savetxt(energy_file, energies)
        pickle.dump({
            'energies': energies,
            'carryover_alpha': carryover_alpha,
            'carryover_beta': carryover_beta,
            'best_energy': best_energy, 'prev_energy': prev_energy,
            'prev_occ': prev_occ, 'occupancies': occupancies,
            'rng': rng, 'next_iter': it + 1,
        }, open(checkpoint_path, 'wb'))
        sys.stdout.flush()

        if converged:
            break

    print(f"\nFinal energy: {energies[-1]:.10f} Ha")
    print(f"Best energy:  {best_energy:.10f} Ha")
    print(f"Iterations:   {len(energies)}")


if __name__ == "__main__":
    main()
