#!/usr/bin/env python3
"""Zero-argument status report for all SQD+SBD experiments.

Run from sqd_sbd/ with no arguments. Auto-discovers all fragments and variants,
reads energy/stats files, and prints a summary table.
"""
import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Standard variant suffixes applied to every fragment.
# Each fragment's results_prefix is prepended to these paths.
# Variants whose directories don't exist yet are silently skipped.
STANDARD_VARIANTS = [
    ('Hardware',          'hardware'),
    ('SC 10k',            'semiclassical/results_10k'),
    ('SC 50k',            'semiclassical/results_50k'),
    ('SC 100k',           'semiclassical/results_100k'),
    ('Random Hamming 50k','semiclassical/results_random_hamming_50k'),
    ('Random IID 50k',    'semiclassical/results_random_iid_50k'),
]

# Fragment definitions
FRAGMENTS = [
    {
        'label': 'ATP f4',
        'norb': 32, 'nelec': 32,
        'base': SCRIPT_DIR,
        'results_prefix': 'f4/results',
    },
    {
        'label': 'ATP f2',
        'norb': 44, 'nelec': 44,
        'base': SCRIPT_DIR,
        'results_prefix': 'f2/results',
        'stale_variants': ['Hardware'],  # old 100k-only data, will be rerun with pooled shots
    },
    {
        'label': 'Metaphosphate',
        'norb': 22, 'nelec': 32,
        'base': os.path.join(SCRIPT_DIR, '..', '..', 'metaphosphate', 'sqd_sbd'),
        'results_prefix': 'results',
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
    """Discover run directories (singleton_, cumulative_, semiclassical_).

    Returns (singletons, cumulatives) where each is a list of (name, path).
    For semiclassical dirs: semiclassical_0 is treated as a singleton,
    semiclassical_0_1_... as cumulatives.
    """
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
        elif name.startswith('semiclassical_'):
            parts = name.split('_')
            nums = [int(p) for p in parts[1:] if p.isdigit()]
            if nums == [0]:
                singletons.append((name, full_path))
            elif len(nums) > 1:
                cumulatives.append((name, full_path))

    # Sort by the numeric part (last number for cumulatives)
    def sort_key(item):
        parts = item[0].split('_')
        nums = [int(p) for p in parts[1:] if p.isdigit()]
        return (len(nums), nums[-1] if nums else 0)

    singletons.sort(key=sort_key)
    cumulatives.sort(key=sort_key)
    return singletons, cumulatives


def print_variant(fragment_label, variant_name, variant_dir, norb, nelec,
                   all_results, stale=False):
    """Print status table for one fragment+variant. Returns (total, converged) counts."""
    singletons, cumulatives = discover_runs(variant_dir)

    stale_tag = " [STALE — will be rerun]" if stale else ""
    print(f"\n=== {fragment_label} ({norb} orb, {nelec} elec) — {variant_name}{stale_tag} ===")

    if not singletons and not cumulatives:
        print(f"  (no runs yet)")
        return 0, 0

    print(f"  {'Config':<45s}  {'Energy (Ha)':>14s}  {'Iters':>6s}  {'Conv':>4s}  "
          f"{'Delta':>10s}  {'Dets':>10s}  {'Wall':>8s}")

    total = 0
    conv = 0
    for name, path in singletons + cumulatives:
        result = load_run(path)
        if result is None:
            continue

        conv_str = "YES" if result['converged'] else "no"
        det_str = f"{result['determinants']:,}" if result['determinants'] else "?"
        wall_str = f"{result['wall_sec']:.0f}s" if result['wall_sec'] else "?"
        delta_str = f"{result['delta']:.2e}" if result['delta'] != float('inf') else "N/A"

        print(f"  {name:<45s}  {result['energy']:>14.4f}  {result['iters']:>6d}  "
              f"{conv_str:>4s}  {delta_str:>10s}  {det_str:>10s}  {wall_str:>8s}")

        result['config'] = name
        result['fragment'] = fragment_label
        result['variant'] = variant_name
        all_results.append(result)
        total += 1
        if result['converged']:
            conv += 1

    return total, conv


def discover_adapt_iters(frag_base, results_prefix):
    """Discover available ADAPT iterations from hardware singleton directories."""
    hw_dir = os.path.join(frag_base, results_prefix, 'hardware')
    adapt_iters = []
    if os.path.isdir(hw_dir):
        for name in os.listdir(hw_dir):
            if name.startswith('singleton_'):
                parts = name.split('_')
                try:
                    adapt_iters.append(int(parts[1]))
                except (ValueError, IndexError):
                    pass
    return sorted(adapt_iters)


def expected_runs(variant_name, n_adapt):
    """Compute expected number of runs for a variant given N available ADAPT iterations.

    Hardware: N singletons + (N-1) cumulatives = 2N - 1
    SC/Random: 1 singleton (SC/random-only) + N cumulatives (adding ADAPT iters one at a time) = N + 1
    """
    if n_adapt == 0:
        return 0
    if variant_name == 'Hardware':
        return 2 * n_adapt - 1
    else:
        # SC and random: semiclassical_0 + semiclassical_0_1 + ... + semiclassical_0_1_..._last
        return n_adapt + 1


def main():
    all_results = []
    # (fragment, variant, total, converged, stale, expected)
    category_stats = []

    for frag in FRAGMENTS:
        base = os.path.realpath(frag['base'])
        prefix = frag['results_prefix']
        stale_variants = frag.get('stale_variants', [])
        adapt_iters = discover_adapt_iters(base, prefix)
        n_adapt = len(adapt_iters)
        if adapt_iters:
            print(f"\n{frag['label']}: {n_adapt} ADAPT iterations available: {adapt_iters}")

        for variant_name, variant_suffix in STANDARD_VARIANTS:
            variant_dir = os.path.join(base, prefix, variant_suffix)
            is_stale = variant_name in stale_variants
            total, conv = print_variant(frag['label'], variant_name, variant_dir,
                                        frag['norb'], frag['nelec'], all_results,
                                        stale=is_stale)
            exp = expected_runs(variant_name, n_adapt)
            category_stats.append((frag['label'], variant_name, total, conv, is_stale, exp))

    # Summary table
    print(f"\n{'='*95}")
    print(f"  {'Fragment':<20s}  {'Variant':<22s}  {'Expect':>6s}  {'Done':>6s}  {'Conv':>6s}  {'Unconv':>6s}  {'Remain':>6s}")
    print(f"  {'-'*20}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    total_exp = 0
    total_done = 0
    total_conv = 0
    total_remain = 0
    for frag_label, variant_name, total, conv, stale, exp in category_stats:
        if stale:
            # Stale = needs full redo
            done = 0
            remain = exp
            status = " STALE"
        else:
            done = total
            remain = max(0, exp - conv)  # unconverged + not yet started
            status = ""
            if total == 0 and exp > 0:
                status = " NOT STARTED"
        unconv = done - conv
        total_exp += exp
        total_done += done
        total_conv += conv
        total_remain += remain
        print(f"  {frag_label:<20s}  {variant_name:<22s}  {exp:>6d}  {done:>6d}  {conv:>6d}  {unconv:>6d}  {remain:>6d}{status}")
    print(f"  {'-'*20}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    print(f"  {'TOTAL':<20s}  {'':<22s}  {total_exp:>6d}  {total_done:>6d}  {total_conv:>6d}  "
          f"{total_done - total_conv:>6d}  {total_remain:>6d}")
    print(f"{'='*95}")
    print(f"  {total_remain} jobs remaining to reach full convergence.")
    print()

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
            'total_expected': total_exp,
            'total_done': total_done,
            'total_converged': total_conv,
            'total_remaining': total_remain,
            'categories': [
                {'fragment': f, 'variant': v, 'expected': e, 'total': t,
                 'converged': c, 'unconverged': t - c, 'stale': s,
                 'remaining': max(0, e - c) if not s else e}
                for f, v, t, c, s, e in category_stats
            ],
            'runs': json_results,
        }, f, indent=2)
    print(f"\nSaved machine-readable status to {status_file}")


if __name__ == "__main__":
    main()
