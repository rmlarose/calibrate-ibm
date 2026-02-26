"""
Unified SQD loop using SBD (GPU) for diagonalization.

Supports two modes:
  - Default (symmetrized spin): alpha/beta CI strings merged into one set
  - --semiclassical_counts: merge pre-generated bitstring counts with hardware data

Replicates the behavior of qiskit-addon-sqd's diagonalize_fermionic_hamiltonian
but uses SBD instead of PySCF for the Selected CI diagonalization.
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
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})

import pyscf.tools

from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample


# ---------- SBD interface ----------

def ci_strings_to_file(ci_strs, norb, filepath):
    """Convert numpy integer CI strings to SBD's binary text format."""
    with open(filepath, 'w') as f:
        for val in ci_strs:
            bits = format(int(val), f'0{norb}b')
            f.write(bits + '\n')


def parse_sbd_output(stdout):
    """Parse energy and separate alpha/beta density from SBD's stdout."""
    energy = None
    density = None
    density_alpha = None
    density_beta = None

    for line in stdout.split('\n'):
        if 'Sample-based diagonalization: Energy =' in line:
            energy = float(line.split('=')[1].strip())
        elif 'Sample-based diagonalization: density_alpha =' in line:
            match = re.search(r'\[(.+)\]', line)
            if match:
                nums = match.group(1).split(',')
                density_alpha = np.array([float(x) for x in nums])
        elif 'Sample-based diagonalization: density_beta =' in line:
            match = re.search(r'\[(.+)\]', line)
            if match:
                nums = match.group(1).split(',')
                density_beta = np.array([float(x) for x in nums])
        elif 'Sample-based diagonalization: density =' in line:
            match = re.search(r'\[(.+)\]', line)
            if match:
                nums = match.group(1).split(',')
                density = np.array([float(x) for x in nums])

    return energy, density, density_alpha, density_beta


def extract_carryover_from_wf(wf_path, adet_path, norb, carryover_threshold):
    """Extract carryover determinants from SBD's wavefunction dump.

    Replicates qiskit-addon-sqd's carryover logic: keep all alpha/beta
    string indices that appear in any amplitude > carryover_threshold.
    """
    # Read the alpha det strings (in order) to map indices back to integers
    adet_strs = []
    with open(adet_path) as f:
        for line in f:
            line = line.strip()
            if line:
                adet_strs.append(line)

    # Parse wf.txt: each line is "amplitude # ia: alpha_bits ib: beta_bits"
    n_a = len(adet_strs)
    # Build amplitude matrix
    amplitudes = {}
    with open(wf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            amp_str, rest = line.split(" # ", 1)
            amp = float(amp_str)
            # Parse "ia: alpha_bits ib: beta_bits"
            parts = rest.split()
            # Format: "0:" "bits" "0:" "bits" or similar
            ia = int(parts[0].rstrip(':'))
            ib = int(parts[2].rstrip(':'))
            amplitudes[(ia, ib)] = amp

    if not amplitudes:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Find all (ia, ib) pairs where |amplitude| > threshold
    alpha_indices = set()
    beta_indices = set()
    for (ia, ib), amp in amplitudes.items():
        if abs(amp) > carryover_threshold:
            alpha_indices.add(ia)
            beta_indices.add(ib)

    # Convert indices to CI string integers
    carryover_a = np.array([int(adet_strs[ia], 2) for ia in sorted(alpha_indices)],
                           dtype=np.int64)
    # For symmetrize_spin, beta = alpha, so beta carryover uses same strings
    carryover_b = np.array([int(adet_strs[ib], 2) for ib in sorted(beta_indices)],
                           dtype=np.int64)

    return carryover_a, carryover_b


def run_sbd(ci_strings_a, ci_strings_b, fcidump_path, sbd_exe, norb,
            carryover_a=None, carryover_b=None,
            carryover_threshold=1e-5,
            block=10, tolerance=1e-6, max_davidson=200,
            num_gpus=1):
    """Run SBD diag executable and return energy + occupancies + carryover."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adet_path = os.path.join(tmpdir, "AlphaDets.txt")
        bdet_path_actual = None

        # Merge carryover strings with sampled strings (carryover first for priority)
        all_a = ci_strings_a
        if carryover_a is not None and len(carryover_a) > 0:
            all_a = np.unique(np.concatenate([carryover_a, ci_strings_a]))

        ci_strings_to_file(all_a, norb, adet_path)

        bdet_args = []
        if ci_strings_b is not None and not np.array_equal(ci_strings_a, ci_strings_b):
            bdet_path_actual = os.path.join(tmpdir, "BetaDets.txt")
            all_b = ci_strings_b
            if carryover_b is not None and len(carryover_b) > 0:
                all_b = np.unique(np.concatenate([carryover_b, ci_strings_b]))
            ci_strings_to_file(all_b, norb, bdet_path_actual)
            bdet_args = ["--bdetfile", bdet_path_actual]

        wf_path = os.path.join(tmpdir, "wf.txt")

        mpirun = os.environ.get("MPIRUN", "mpirun")
        cmd = [
            mpirun, "--oversubscribe", "-np", str(num_gpus),
            sbd_exe,
            "--fcidump", fcidump_path,
            "--adetfile", adet_path,
            *bdet_args,
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
            raise RuntimeError(f"SBD failed with return code {result.returncode}")

        energy, density, density_alpha, density_beta = parse_sbd_output(result.stdout)
        if energy is None:
            print(f"SBD STDOUT: {result.stdout}", file=sys.stderr)
            raise RuntimeError("Failed to parse energy from SBD output")

        # Extract carryover from CI amplitudes
        co_a, co_b = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        if os.path.exists(wf_path):
            co_a, co_b = extract_carryover_from_wf(
                wf_path, adet_path, norb, carryover_threshold)

        # Use separate alpha/beta occupancies if available, else fall back to density/2
        if density_alpha is not None and density_beta is not None:
            occ_alpha = density_alpha
            occ_beta = density_beta
        elif density is not None:
            occ_alpha = density / 2.0
            occ_beta = density / 2.0
        else:
            occ_alpha = None
            occ_beta = None

        return energy, density, occ_alpha, occ_beta, co_a, co_b


# ---------- Bitstring processing (from isqd.py) ----------

def transform_bitstring(bits):
    """Openfermion convention -> Qiskit convention."""
    left = [bits[i] for i in range(len(bits)) if i % 2 == 1]   # beta
    right = [bits[i] for i in range(len(bits)) if i % 2 == 0]  # alpha
    left.reverse()
    right.reverse()
    return ''.join(left + right)


def load_and_process_bitstrings(fragment, circuit_dir, hamiltonian_dir, results_dir,
                                 adapt_iterations, n_orbitals, num_electrons):
    """Load measurement data and apply qubit/mode permutations."""
    nqubits = 2 * n_orbitals
    all_counts = []

    for adapt_iter in adapt_iterations:
        fnames = sorted(glob.glob(f"{results_dir}/{fragment}/*{adapt_iter:03d}*.qasm*"))
        if not fnames:
            raise FileNotFoundError(
                f"No bitstring files found for ADAPT iter {adapt_iter} "
                f"in {results_dir}/{fragment}/")
        mode_order = pickle.load(
            open(f"{circuit_dir}/{fragment}/{fragment}_mode_order_{adapt_iter:03d}_adaptiterations.pkl", "rb")
        )
        qubit_order = pickle.load(
            open(f"{circuit_dir}/{fragment}/{fragment}_qubit_order_{adapt_iter:03d}_adaptiterations.pkl", "rb")
        )

        # Load and merge ALL matching files for this ADAPT iteration
        iter_total_unique = 0
        iter_total_shots = 0
        permuted = {}
        for fname in fnames:
            counts = pickle.load(open(fname, "rb"))
            iter_total_unique += len(counts)
            iter_total_shots += sum(counts.values())
            for orig_bs in counts.keys():
                qp = "".join([orig_bs[qubit_order.index(n)] for n in range(nqubits)])
                mp = "".join([qp[mode_order.index(n)] for n in range(nqubits)])
                final = transform_bitstring(mp)
                permuted[final] = permuted.get(final, 0) + counts[orig_bs]

        all_counts.append(permuted)
        n_files = len(fnames)
        files_str = f" ({n_files} files)" if n_files > 1 else ""
        print(f"  ADAPT iter {adapt_iter}: {iter_total_unique} raw unique, "
              f"{iter_total_shots} shots{files_str} -> {len(permuted)} merged unique")

    # Merge
    merged = collections.Counter()
    for c in all_counts:
        for bs, count in c.items():
            merged[bs] += count

    return merged


def bitstrings_to_matrix(counts, nqubits):
    """Convert counts dict to (bitstring_matrix, probabilities)."""
    total = sum(counts.values())
    bitstrings = []
    probs = []
    for bs, count in counts.items():
        row = np.array([int(b) for b in bs], dtype=np.bool_)
        bitstrings.append(row)
        probs.append(count / total)

    return np.array(bitstrings), np.array(probs)


def bitstring_matrix_to_integers(matrix):
    """Convert a 2D bool matrix to 1D array of integers (big-endian)."""
    n = matrix.shape[1]
    powers = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    return matrix.astype(np.int64) @ powers


def extract_ci_strings(batch, n_orbitals):
    """Extract unique alpha CI strings from a batch of bitstrings.
    With symmetrize_spin, alpha and beta share the same string set.
    Returns sorted numpy array of integer CI strings."""
    alpha_part = batch[:, n_orbitals:]  # right half
    beta_part = batch[:, :n_orbitals]   # left half

    alpha_ints = set(bitstring_matrix_to_integers(alpha_part).tolist())
    beta_ints = set(bitstring_matrix_to_integers(beta_part).tolist())

    # Symmetrize: merge alpha and beta
    all_strings = alpha_ints | beta_ints
    return np.array(sorted(all_strings), dtype=np.int64)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Full SQD loop with SBD GPU solver")
    parser.add_argument("--fragment", type=str, default="atp_0_be2_f4")
    parser.add_argument("--circuit_dir", type=str, default="../circuits")
    parser.add_argument("--hamiltonian_dir", type=str, default="../hamiltonians")
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--sbd_exe", type=str, required=True,
                        help="Path to SBD diag executable")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--samples_per_batch", type=int, default=500)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--energy_tol", type=float, default=1e-8)
    parser.add_argument("--occupancies_tol", type=float, default=1e-8)
    parser.add_argument("--carryover_threshold", type=float, default=1e-5)
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for SBD (MPI ranks)")
    parser.add_argument("--semiclassical_counts", type=str, default=None,
                        help="Path to pre-generated semiclassical counts pickle")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("adapt_iterations", type=int, nargs='*', default=[1, 2, 3, 4, 5, 10, 20, 25, 30, 40])
    args = parser.parse_args()

    if not args.semiclassical_counts and not args.adapt_iterations:
        parser.error("Must provide --semiclassical_counts and/or ADAPT iteration indices")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build iteration key (include "0" for semiclassical)
    all_iters = []
    if args.semiclassical_counts:
        all_iters.append(0)
    all_iters.extend(args.adapt_iterations)
    iters_key = "_".join(map(str, all_iters))
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{iters_key}.pkl")
    energy_file = os.path.join(args.output_dir, f"sqd_energies_{iters_key}.txt")
    occ_file = os.path.join(args.output_dir, f"sqd_occupancies_{iters_key}.txt")
    stats_file = os.path.join(args.output_dir, f"sqd_stats_{iters_key}.txt")
    plot_file = os.path.join(args.output_dir, f"sqd_convergence_{iters_key}.pdf")

    # Read Hamiltonian
    print("Reading Hamiltonian...")
    fcidump_path = f"{args.hamiltonian_dir}/{args.fragment}.fcidump"
    fcidump = pyscf.tools.fcidump.read(fcidump_path)
    n_orbitals = fcidump.get("NORB")
    num_electrons = fcidump.get("NELEC")
    ecore = fcidump.get("ECORE")
    n_alpha = num_electrons // 2
    n_beta = num_electrons // 2
    print(f"  NORB={n_orbitals}, NELEC={num_electrons}, ECORE={ecore}")

    # Load measurement data (raw, before postselection)
    merged_counts = collections.Counter()

    if args.semiclassical_counts:
        print(f"Loading semiclassical counts from {args.semiclassical_counts}...")
        with open(args.semiclassical_counts, 'rb') as f:
            sc_counts = pickle.load(f)
        for bs, count in sc_counts.items():
            merged_counts[bs] += count
        print(f"  Semiclassical: {len(sc_counts)} unique, {sum(sc_counts.values())} shots")

    if args.adapt_iterations:
        print("Loading measurement data...")
        hw_counts = load_and_process_bitstrings(
            args.fragment, args.circuit_dir, args.hamiltonian_dir,
            args.results_dir, args.adapt_iterations, n_orbitals, num_electrons
        )
        for bs, count in hw_counts.items():
            merged_counts[bs] += count

    print(f"  Total: {len(merged_counts)} unique bitstrings, {sum(merged_counts.values())} shots")

    # Convert to matrix form - keep RAW bitstrings for config recovery
    raw_bs_matrix, raw_probs = bitstrings_to_matrix(merged_counts, 2 * n_orbitals)
    print(f"  Raw bitstring matrix: {raw_bs_matrix.shape}")

    # SQD Loop
    print(f"\n{'='*60}")
    print(f"Starting SQD loop: {args.num_batches} batches, {args.samples_per_batch} samples/batch")
    print(f"Max iterations: {args.max_iterations}")
    print(f"{'='*60}")
    sys.stdout.flush()

    rng = np.random.default_rng(42)
    current_occupancies = None
    carryover_a = None
    carryover_b = None
    best_ever_energy = None
    current_energy = None
    current_occ = None
    converged = False

    iteration_energies = []
    all_occupancies = []
    iteration_stats = []  # per-iteration: {ci_strings, determinants, pool_size, carryover, wall_sec}
    start_iteration = 0

    # Resume from checkpoint if available
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        iteration_energies = ckpt['iteration_energies']
        all_occupancies = ckpt['all_occupancies']
        carryover_a = ckpt['carryover_a']
        carryover_b = ckpt['carryover_b']
        best_ever_energy = ckpt['best_ever_energy']
        current_energy = ckpt['current_energy']
        current_occ = ckpt['current_occ']
        current_occupancies = ckpt['current_occupancies']
        rng = ckpt['rng']
        start_iteration = ckpt['next_iteration']
        iteration_stats = ckpt.get('iteration_stats', [])
        print(f"  Resuming at iteration {start_iteration + 1}, "
              f"best energy so far: {best_ever_energy:.10f} Ha")

    for iteration in range(start_iteration, args.max_iterations):
        iter_t0 = time.time()
        print(f"\n--- SQD Iteration {iteration + 1} ---")
        sys.stdout.flush()

        # Step 1: Postselect or Recover configurations
        # Key: always operate on the RAW bitstrings, not pre-filtered ones
        if current_occupancies is not None:
            # Config recovery: fixes bitstrings to match expected occupancies
            # and guarantees correct Hamming weight
            bitstrings, probs = recover_configurations(
                raw_bs_matrix, raw_probs, current_occupancies,
                n_alpha, n_beta, rand_seed=rng
            )
        else:
            # First iteration: just postselect by electron count
            bitstrings, probs = postselect_by_hamming_right_and_left(
                raw_bs_matrix, raw_probs,
                hamming_right=n_alpha, hamming_left=n_beta
            )

        pool_size = len(bitstrings)
        print(f"  After postselect/recovery: {pool_size} unique bitstrings")

        # Step 2: Subsample batches
        batches = subsample(
            bitstrings, probs,
            samples_per_batch=args.samples_per_batch,
            num_batches=args.num_batches,
            rand_seed=rng,
        )

        # Step 3: Diagonalize each batch via SBD
        batch_energies = []
        batch_occupancies = []
        batch_carryover = []
        batch_ci_counts = []

        for b, batch in enumerate(batches):
            ci_strings = extract_ci_strings(batch, n_orbitals)
            ci_alpha = ci_strings
            ci_beta = None
            n_dets = len(ci_strings) ** 2
            batch_ci_counts.append(len(ci_strings))
            print(f"  Batch {b}: {len(ci_strings)} unique CI strings "
                  f"({n_dets} determinants)")
            sys.stdout.flush()

            energy, density, occ_alpha, occ_beta, co_a, co_b = run_sbd(
                ci_strings_a=ci_alpha,
                ci_strings_b=ci_beta,
                fcidump_path=fcidump_path,
                sbd_exe=args.sbd_exe,
                norb=n_orbitals,
                carryover_a=carryover_a,
                carryover_b=carryover_b,
                carryover_threshold=args.carryover_threshold,
                block=10,
                tolerance=1e-9,
                max_davidson=10000,
                num_gpus=args.num_gpus,
            )

            total_energy = energy + ecore
            print(f"  Batch {b}: energy = {total_energy:.10f}")
            batch_energies.append(total_energy)
            batch_occupancies.append((occ_alpha, occ_beta))
            batch_carryover.append((co_a, co_b))

        # Step 4: Find best batch in this iteration
        best_idx = np.argmin(batch_energies)
        best_energy = batch_energies[best_idx]
        best_occ_alpha, best_occ_beta = batch_occupancies[best_idx]
        best_ci_count = batch_ci_counts[best_idx]
        iteration_energies.append(best_energy)
        all_occupancies.append((best_occ_alpha + best_occ_beta) / 2.0)  # store avg for plotting

        print(f"  Best energy: {best_energy:.10f} Ha")

        # Update carryover from best batch's CI amplitudes
        co_a, co_b = batch_carryover[best_idx]
        if len(co_a) > 0 or len(co_b) > 0:
            # Symmetrize: merge alpha and beta carryover
            all_co = [x for x in [co_a, co_b] if len(x) > 0]
            carryover_a = np.unique(np.concatenate(all_co))
            carryover_b = carryover_a
            print(f"  Carryover: {len(carryover_a)} strings")

        n_carryover = len(carryover_a) if carryover_a is not None else 0

        # Track best-ever energy
        if best_ever_energy is None or best_energy < best_ever_energy:
            best_ever_energy = best_energy

        # Step 5: Update occupancies for next iteration's config recovery
        current_occupancies = (best_occ_alpha, best_occ_beta)

        # Step 6: Check convergence (compare to previous iteration)
        converged = False
        if current_energy is not None:
            energy_diff = abs(current_energy - best_energy)
            occ_diff = np.linalg.norm(
                np.concatenate([current_occ[0], current_occ[1]])
                - np.concatenate([best_occ_alpha, best_occ_beta]),
                ord=np.inf
            )
            print(f"  Energy change: {energy_diff:.2e}, Max occ change: {occ_diff:.2e}")

            if energy_diff < args.energy_tol and occ_diff < args.occupancies_tol:
                print(f"\n  CONVERGED at iteration {iteration + 1}!")
                converged = True

        current_energy = best_energy
        current_occ = (best_occ_alpha, best_occ_beta)

        # Record per-iteration stats
        iter_wall_sec = time.time() - iter_t0
        n_dets_approx = best_ci_count ** 2
        stats = {
            'iteration': iteration + 1,
            'energy': best_energy,
            'ci_strings': best_ci_count,
            'determinants': n_dets_approx,
            'pool_size': pool_size,
            'carryover': n_carryover,
            'wall_sec': iter_wall_sec,
        }
        iteration_stats.append(stats)
        print(f"  Wall time: {iter_wall_sec:.1f}s | CI strings: {best_ci_count} | "
              f"Dets: {n_dets_approx} | Pool: {pool_size} | Carryover: {n_carryover}")

        # Checkpoint: save state after each iteration
        np.savetxt(energy_file, iteration_energies)
        np.savetxt(occ_file, all_occupancies[-1])

        # Write stats file (rewrite each iteration for consistency)
        with open(stats_file, 'w') as f:
            f.write(f"# {'iter':>4s}  {'energy':>18s}  {'ci_strings':>10s}  {'determinants':>12s}  "
                    f"{'pool_size':>10s}  {'carryover':>10s}  {'wall_sec':>10s}\n")
            for s in iteration_stats:
                f.write(f"  {s['iteration']:>4d}  {s['energy']:>18.10f}  {s['ci_strings']:>10d}  "
                        f"{s['determinants']:>12d}  {s['pool_size']:>10d}  {s['carryover']:>10d}  "
                        f"{s['wall_sec']:>10.1f}\n")

        ckpt = {
            'iteration_energies': iteration_energies,
            'all_occupancies': all_occupancies,
            'iteration_stats': iteration_stats,
            'carryover_a': carryover_a,
            'carryover_b': carryover_b,
            'best_ever_energy': best_ever_energy,
            'current_energy': current_energy,
            'current_occ': current_occ,
            'current_occupancies': current_occupancies,
            'rng': rng,
            'next_iteration': iteration + 1,
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved (iteration {iteration + 1})")
        sys.stdout.flush()

        if converged:
            break

    # Final results summary
    total_wall = sum(s['wall_sec'] for s in iteration_stats)
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    for i, e in enumerate(iteration_energies):
        wall = iteration_stats[i]['wall_sec'] if i < len(iteration_stats) else 0
        print(f"  Iteration {i+1}: {e:.10f} Ha  ({wall:.1f}s)")
    print(f"\n  Best ever: {best_ever_energy:.10f} Ha")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/60:.1f}min)")

    np.savetxt(energy_file, iteration_energies)
    print(f"\nSaved energies to {energy_file}")

    np.savetxt(occ_file, all_occupancies[-1])
    print(f"Saved occupancies to {occ_file}")

    print(f"Saved stats to {stats_file}")

    # Generate publication-quality plots (matching repo style)
    chem_accuracy = 0.001  # 1 mHa

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left panel: Energy convergence (semilogy of |Delta E|)
    x1 = np.arange(1, len(iteration_energies) + 1)
    e_arr = np.array(iteration_energies)

    # Plot energy error relative to best energy achieved
    e_diff = np.abs(e_arr - best_ever_energy)
    e_diff[e_diff == 0] = 1e-12  # avoid log(0)

    axs[0].plot(x1, e_diff, "--o", mec="black", ms=10, alpha=0.75,
                label=r"$| \Delta E |$")
    axs[0].axhline(y=chem_accuracy, color="black", linestyle="--",
                   label="Chemical Accuracy")
    axs[0].set_yscale("log")
    axs[0].set_yticks([1.0, 1e-1, 1e-2, 1e-3, 1e-4])
    axs[0].set_xlabel("SQD Iteration", fontsize=12)
    axs[0].set_ylabel("Energy Error (Ha)", fontsize=12)
    axs[0].set_title(f"SQD Convergence: {args.fragment}", fontsize=12)
    axs[0].legend(fontsize=10)

    # Right panel: Orbital occupancy bar chart
    final_occ = all_occupancies[-1] * 2  # total (alpha + beta)
    x2 = np.arange(len(final_occ))
    axs[1].bar(x2, final_occ, width=0.8)
    axs[1].set_xlabel("Orbital Index", fontsize=12)
    axs[1].set_ylabel("Avg Occupancy", fontsize=12)
    axs[1].set_title("Avg Occupancy per Spatial Orbital", fontsize=12)

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")

    # Always keep checkpoint for safety
    if os.path.exists(checkpoint_path):
        if converged:
            print(f"Checkpoint retained (converged at iteration {len(iteration_energies)})")
        else:
            print(f"Checkpoint retained (not converged after {len(iteration_energies)} iterations)")

    print(f"\nFinal energy: {iteration_energies[-1]:.10f} Ha")
    print(f"Best ever energy: {best_ever_energy:.10f} Ha")
    print(f"Total SQD iterations: {len(iteration_energies)}")


if __name__ == "__main__":
    main()
