#!/usr/bin/env python3
"""Zero-argument status report for all SQD+SBD experiments.

Run from sqd_sbd/ with no arguments. Auto-discovers all fragments and variants,
reads energy/stats files, and prints a summary table.
"""
import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fragment definitions: (label, path_relative_to_script, norb, nelec, variants)
FRAGMENTS = [
    {
        'label': 'ATP f4',
        'norb': 32, 'nelec': 32,
        'base': SCRIPT_DIR,
        'variants': [
            ('Hardware', 'f4/results/hardware'),
            ('Nosym', 'f4/results/nosym'),
            ('Semiclassical 10k', 'f4/results/semiclassical/results_10k'),
            ('Semiclassical 50k', 'f4/results/semiclassical/results_50k'),
            ('Semiclassical 100k', 'f4/results/semiclassical/results_100k'),
            ('Random Hamming 50k', 'f4/results/semiclassical/results_random_hamming_50k'),
            ('Random IID 50k', 'f4/results/semiclassical/results_random_iid_50k'),
        ],
    },
    {
        'label': 'ATP f2',
        'norb': 44, 'nelec': 44,
        'base': SCRIPT_DIR,
        'variants': [
            ('Hardware', 'f2/results/hardware'),
        ],
    },
    {
        'label': 'Metaphosphate',
        'norb': 22, 'nelec': 32,
        'base': os.path.join(SCRIPT_DIR, '..', '..', 'metaphosphate', 'sqd_sbd'),
        'variants': [
            ('Hardware', 'results/hardware'),
        ],
    },
]

CONVERGENCE_THRESHOLD = 1e-7


def load_run(run_dir):
    """Load results from a single SQD run directory.

    Returns dict with keys: energy, iters, converged, delta, ci_strings,
    determinants, wall_sec, or None if no results found.
    """
    if not os.path.isdir(run_dir):
        return None

    energy_files = [f for f in os.listdir(run_dir) if f.startswith("sqd_energies_")]
    if not energy_files:
        return None

    energy_file = os.path.join(run_dir, energy_files[0])
    try:
        energies = np.loadtxt(energy_file)
    except Exception:
        return None
    if energies.ndim == 0:
        energies = np.array([float(energies)])
    if len(energies) == 0:
        return None

    num_iters = len(energies)
    best_energy = float(np.min(energies))

    # Convergence: last two energies within threshold
    delta = 0.0
    converged = False
    if num_iters >= 2:
        delta = abs(energies[-1] - energies[-2])
        converged = delta < CONVERGENCE_THRESHOLD
    elif num_iters == 1:
        delta = float('inf')

    # Load stats for determinant counts and wall time
    stats_files = [f for f in os.listdir(run_dir) if f.startswith("sqd_stats_")]
    ci_strings = None
    determinants = None
    total_wall = None
    if stats_files:
        stats_path = os.path.join(run_dir, stats_files[0])
        try:
            data = np.loadtxt(stats_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            ci_strings = int(data[-1, 2])
            determinants = int(data[-1, 3])
            total_wall = float(np.sum(data[:, 6]))
        except Exception:
            pass

    return {
        'energy': best_energy,
        'iters': num_iters,
        'converged': converged,
        'delta': delta,
        'ci_strings': ci_strings,
        'determinants': determinants,
        'wall_sec': total_wall,
    }


def discover_runs(variant_dir):
    """Discover singleton and cumulative run directories."""
    if not os.path.isdir(variant_dir):
        return [], []

    singletons = []
    cumulatives = []

    for name in sorted(os.listdir(variant_dir)):
        full_path = os.path.join(variant_dir, name)
        if not os.path.isdir(full_path):
            continue
        if name.startswith('singleton_'):
            singletons.append((name, full_path))
        elif name.startswith('cumulative_'):
            cumulatives.append((name, full_path))

    # Sort by the numeric part (last number for cumulatives)
    def sort_key(item):
        parts = item[0].split('_')
        nums = [int(p) for p in parts[1:] if p.isdigit()]
        return (len(nums), nums[-1] if nums else 0)

    singletons.sort(key=sort_key)
    cumulatives.sort(key=sort_key)
    return singletons, cumulatives


def print_variant(fragment_label, variant_name, variant_dir, norb, nelec, all_results):
    """Print status table for one fragment+variant. Returns list of result dicts."""
    singletons, cumulatives = discover_runs(variant_dir)
    if not singletons and not cumulatives:
        return []

    print(f"\n=== {fragment_label} ({norb} orb, {nelec} elec) — {variant_name} ===")
    print(f"  {'Config':<40s}  {'Energy (Ha)':>14s}  {'Iters':>6s}  {'Conv':>4s}  "
          f"{'Delta':>10s}  {'Dets':>10s}  {'Wall':>8s}")

    results = []
    for name, path in singletons + cumulatives:
        result = load_run(path)
        if result is None:
            continue

        conv_str = "YES" if result['converged'] else "no"
        det_str = f"{result['determinants']:,}" if result['determinants'] else "?"
        wall_str = f"{result['wall_sec']:.0f}s" if result['wall_sec'] else "?"
        delta_str = f"{result['delta']:.2e}" if result['delta'] != float('inf') else "N/A"

        print(f"  {name:<40s}  {result['energy']:>14.4f}  {result['iters']:>6d}  "
              f"{conv_str:>4s}  {delta_str:>10s}  {det_str:>10s}  {wall_str:>8s}")

        result['config'] = name
        result['fragment'] = fragment_label
        result['variant'] = variant_name
        results.append(result)
        all_results.append(result)

    return results


def main():
    all_results = []

    for frag in FRAGMENTS:
        base = os.path.realpath(frag['base'])
        for variant_name, variant_rel in frag['variants']:
            # Handle semiclassical subdirectory case:
            # the runs may be directly in the variant dir (not in singleton_*/cumulative_*)
            variant_dir = os.path.join(base, variant_rel)
            if not os.path.isdir(variant_dir):
                continue

            # Check if this dir has singleton_/cumulative_ subdirs
            entries = os.listdir(variant_dir) if os.path.isdir(variant_dir) else []
            has_run_dirs = any(e.startswith(('singleton_', 'cumulative_'))
                              for e in entries)

            if has_run_dirs:
                print_variant(frag['label'], variant_name, variant_dir,
                              frag['norb'], frag['nelec'], all_results)
            else:
                # Semiclassical: results may be directly in this dir
                result = load_run(variant_dir)
                if result is not None:
                    print(f"\n=== {frag['label']} ({frag['norb']} orb, {frag['nelec']} elec) — {variant_name} ===")
                    conv_str = "YES" if result['converged'] else "no"
                    det_str = f"{result['determinants']:,}" if result['determinants'] else "?"
                    wall_str = f"{result['wall_sec']:.0f}s" if result['wall_sec'] else "?"
                    delta_str = f"{result['delta']:.2e}" if result['delta'] != float('inf') else "N/A"
                    print(f"  {'(single run)':<40s}  {result['energy']:>14.4f}  {result['iters']:>6d}  "
                          f"{conv_str:>4s}  {delta_str:>10s}  {det_str:>10s}  {wall_str:>8s}")
                    result['config'] = variant_name
                    result['fragment'] = frag['label']
                    result['variant'] = variant_name
                    all_results.append(result)

    # Summary
    total = len(all_results)
    converged = sum(1 for r in all_results if r['converged'])
    unconverged = total - converged
    print(f"\n{'='*80}")
    print(f"SUMMARY: {total} total runs | {converged} converged | {unconverged} in progress/unconverged")
    print(f"{'='*80}")

    # Write machine-readable JSON
    status_file = os.path.join(SCRIPT_DIR, "status.json")
    # Convert for JSON serialization
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (np.bool_, np.integer)):
                jr[k] = int(v) if isinstance(v, np.integer) else bool(v)
            elif isinstance(v, np.floating):
                jr[k] = float(v)
            elif v == float('inf'):
                jr[k] = None
            else:
                jr[k] = v
        if jr.get('delta') == float('inf'):
            jr['delta'] = None
        json_results.append(jr)
    with open(status_file, 'w') as f:
        json.dump({
            'total': total,
            'converged': converged,
            'unconverged': unconverged,
            'runs': json_results,
        }, f, indent=2)
    print(f"\nSaved machine-readable status to {status_file}")


if __name__ == "__main__":
    main()
