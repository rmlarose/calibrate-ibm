#!/usr/bin/env python3
"""Zero-argument plot generation for all SQD+SBD experiments.

Run from sqd_sbd/ with no arguments. Auto-discovers all fragments and variants,
generates energy/iteration/determinant plots for each.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})
import pyscf.tools.fcidump

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CCSD energy files: maps fragment_name -> file path
# Each file has columns: System, SCF, MP2, CCSD, CCSD(T)
CCSD_FILES = [
    os.path.join(SCRIPT_DIR, '..', 'atp-fragment-ccsd-energies.txt'),
]


def load_ccsd_energies():
    """Load CCSD energies from reference files.

    Returns dict mapping fragment_name -> {'SCF': ..., 'MP2': ..., 'CCSD': ..., 'CCSD(T)': ...}.
    """
    energies = {}
    for fpath in CCSD_FILES:
        fpath = os.path.realpath(fpath)
        if not os.path.exists(fpath):
            print(f"  Warning: CCSD file not found: {fpath}")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    name = parts[0]
                    energies[name] = {
                        'SCF': float(parts[1]),
                        'MP2': float(parts[2]),
                        'CCSD': float(parts[3]),
                        'CCSD(T)': float(parts[4]),
                    }
    return energies


# Fragment definitions
FRAGMENTS = [
    {
        'label': 'ATP f4',
        'fragment_name': 'atp_0_be2_f4',
        'norb': 32, 'nelec': 32,
        'base': SCRIPT_DIR,
        'plots_dir': 'f4/plots',
        'hamiltonian': os.path.join(SCRIPT_DIR, '..', 'hamiltonians', 'atp_0_be2_f4.fcidump'),
        'variants': [
            ('Hardware', 'f4/results/hardware'),
        ],
        'sc_variants': [
            ('SC 10k', 'f4/results/semiclassical/results_10k'),
            ('SC 50k', 'f4/results/semiclassical/results_50k'),
            ('SC 100k', 'f4/results/semiclassical/results_100k'),
        ],
        'random_variants': [
            ('Random Hamming 50k', 'f4/results/semiclassical/results_random_hamming_50k'),
            ('Random IID 50k', 'f4/results/semiclassical/results_random_iid_50k'),
        ],
    },
    {
        'label': 'ATP f2',
        'fragment_name': 'atp_0_be2_f2',
        'norb': 44, 'nelec': 44,
        'base': SCRIPT_DIR,
        'plots_dir': 'f2/plots',
        'hamiltonian': os.path.join(SCRIPT_DIR, '..', 'hamiltonians', 'atp_0_be2_f2.fcidump'),
        'variants': [
            ('Hardware', 'f2/results/hardware'),
        ],
        'sc_variants': [],
        'random_variants': [],
    },
    {
        'label': 'Metaphosphate',
        'fragment_name': 'metaphosphate-2026',
        'norb': 22, 'nelec': 32,
        'base': os.path.join(SCRIPT_DIR, '..', '..', 'metaphosphate', 'sqd_sbd'),
        'plots_dir': 'plots',
        'hamiltonian': os.path.join(SCRIPT_DIR, '..', '..', 'metaphosphate', 'hamiltonians',
                                    'metaphosphate-2026.fcidump'),
        'variants': [
            ('Hardware', 'results/hardware'),
        ],
        'sc_variants': [],
        'random_variants': [],
    },
]


def compute_hf_energy(fcidump_path):
    """Compute HF energy from FCIDUMP file."""
    if not os.path.exists(fcidump_path):
        return None
    try:
        fcid = pyscf.tools.fcidump.read(fcidump_path)
        mf = pyscf.tools.fcidump.to_scf(fcidump_path)
        mf.kernel()
        return mf.e_tot + fcid['ECORE']
    except Exception as e:
        print(f"  Warning: could not compute HF energy from {fcidump_path}: {e}")
        return None


def load_run(run_dir):
    """Load results from a single SQD run directory."""
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

    converged = False
    if num_iters >= 2:
        converged = abs(energies[-1] - energies[-2]) < 1e-7

    # Load stats
    stats_files = [f for f in os.listdir(run_dir) if f.startswith("sqd_stats_")]
    determinants = None
    if stats_files:
        try:
            data = np.loadtxt(os.path.join(run_dir, stats_files[0]))
            if data.ndim == 1:
                data = data.reshape(1, -1)
            determinants = int(data[-1, 3])
        except Exception:
            pass

    return {
        'energy': best_energy,
        'iters': num_iters,
        'converged': converged,
        'determinants': determinants,
    }


def discover_and_load(variant_dir):
    """Discover and load all singleton/cumulative runs in a variant directory.

    Returns (singletons_dict, cumulatives_dict) where keys are the "last ADAPT iter"
    integer and values are result dicts.
    """
    singletons = {}
    cumulatives = {}

    if not os.path.isdir(variant_dir):
        return singletons, cumulatives

    for name in os.listdir(variant_dir):
        full_path = os.path.join(variant_dir, name)
        if not os.path.isdir(full_path):
            continue

        result = load_run(full_path)
        if result is None:
            continue

        if name.startswith('singleton_'):
            parts = name.split('_')
            try:
                adapt = int(parts[1])
                singletons[adapt] = result
            except (ValueError, IndexError):
                pass
        elif name.startswith('cumulative_'):
            parts = name.split('_')
            nums = [int(p) for p in parts[1:] if p.isdigit()]
            if nums:
                last_adapt = nums[-1]
                cumulatives[last_adapt] = result

    return singletons, cumulatives


def discover_and_load_sc(variant_dir):
    """Discover and load SC/random runs from a semiclassical variant directory.

    These use naming like semiclassical_0 (SC-only), semiclassical_0_1 (SC + ADAPT 1),
    semiclassical_0_1_2 (SC + ADAPT 1+2), etc.

    Returns (singleton_result_or_None, runs_dict) where:
    - singleton is the semiclassical_0 run (SC/random only, no hardware)
    - runs_dict is keyed by last ADAPT iter (including 0 for singleton) -> result dict
    """
    singleton = None
    runs = {}
    if not os.path.isdir(variant_dir):
        return singleton, runs

    for name in os.listdir(variant_dir):
        full_path = os.path.join(variant_dir, name)
        if not os.path.isdir(full_path) or not name.startswith('semiclassical_'):
            continue

        result = load_run(full_path)
        if result is None:
            continue

        parts = name.split('_')
        nums = [int(p) for p in parts[1:] if p.isdigit()]
        if nums == [0]:
            singleton = result
            runs[0] = result
        elif len(nums) > 1:
            last_adapt = nums[-1]
            runs[last_adapt] = result

    return singleton, runs


def plot_series(ax, data_dict, color, marker, label, x_keys=None, yidx='energy'):
    """Plot a data series with stars for non-converged points."""
    if x_keys is None:
        x_keys = sorted(data_dict.keys())
    all_x = [x for x in x_keys if x in data_dict]
    all_y = [data_dict[x][yidx] for x in all_x]
    conv_x = [x for x in all_x if data_dict[x]['converged']]
    conv_y = [data_dict[x][yidx] for x in conv_x]
    prog_x = [x for x in all_x if not data_dict[x]['converged']]
    prog_y = [data_dict[x][yidx] for x in prog_x]

    if all_x:
        ax.plot(all_x, all_y, '-', color=color, zorder=1)
    if conv_x:
        ax.plot(conv_x, conv_y, marker, color=color, ms=8, mec=color,
                label=label, zorder=2)
    if prog_x:
        star_label = f'{label} (in progress)' if not conv_x else None
        ax.plot(prog_x, prog_y, '*', color=color, ms=14, mec=color,
                alpha=0.5, label=star_label, zorder=3)


def add_ref_lines(ax, ref_lines):
    """Add labeled reference lines to the y-axis."""
    if not ref_lines:
        return
    ref_vals = [r[0] for r in ref_lines]
    default_ticks = [t for t in ax.get_yticks()
                     if all(abs(t - r) > 0.02 for r in ref_vals)]
    yticks = list(default_ticks) + ref_vals
    ytick_labels = [f'{t:.2f}' for t in default_ticks] + [r[2] for r in ref_lines]
    ytick_colors = ['black'] * len(default_ticks) + [r[1] for r in ref_lines]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    for tick_label, color in zip(ax.get_yticklabels(), ytick_colors):
        tick_label.set_color(color)


def make_energy_plot(ax, hw_variants, sc_series, adapt_iters, e_hf, e_ccsd,
                     e_random_best, e_best_sc, title):
    """Build an energy-vs-ADAPT plot on the given axes.

    hw_variants: dict of variant_name -> (singletons, cumulatives) for hardware
    sc_series: list of (label, color, runs_dict) for SC/random cumulative series
    e_best_sc: best SC-only energy (from SC cumulatives, not random), or None
    """
    # Compute best hardware-only energy
    hw_energies = []
    for singletons, cumulatives in hw_variants.values():
        hw_energies.extend(v['energy'] for v in singletons.values())
        hw_energies.extend(v['energy'] for v in cumulatives.values())
    e_best_hw = min(hw_energies) if hw_energies else None

    # Compute y-axis limits from hardware data + reference lines before plotting
    ylim_vals = list(hw_energies)
    for e in [e_hf, e_ccsd, e_random_best, e_best_sc]:
        if e is not None:
            ylim_vals.append(e)
    if ylim_vals:
        margin = (max(ylim_vals) - min(ylim_vals)) * 0.08
        ylim_lo = min(ylim_vals) - margin
        ylim_hi = max(ylim_vals) + margin

    # Hardware data
    hw_colors = {
        'Hardware': ('tab:blue', 'tab:orange'),
    }
    for variant_name, (singletons, cumulatives) in hw_variants.items():
        sc, cc = hw_colors.get(variant_name, ('tab:blue', 'tab:orange'))
        suffix = f' ({variant_name})' if len(hw_variants) > 1 else ''
        plot_series(ax, singletons, sc, 'o', f'Singleton{suffix}', adapt_iters, 'energy')
        plot_series(ax, cumulatives, cc, 's', f'Cumulative{suffix}', yidx='energy')

    # SC/random cumulative series
    for label, color, runs in sc_series:
        plot_series(ax, runs, color, 'D', label, yidx='energy')

    # Lock y-axis to hardware range (outlier SC/random points will be clipped)
    if ylim_vals:
        ax.set_ylim(ylim_lo, ylim_hi)

    # Reference lines
    if e_hf is not None:
        ax.axhline(y=e_hf, color='red', linestyle='--')
    if e_ccsd is not None:
        ax.axhline(y=e_ccsd, color='tab:green', linestyle='--')
    if e_random_best is not None:
        ax.axhline(y=e_random_best, color='gray', linestyle='--', alpha=0.7)
    if e_best_hw is not None:
        ax.axhline(y=e_best_hw, color='tab:orange', linestyle=':', alpha=0.7)
    if e_best_sc is not None:
        ax.axhline(y=e_best_sc, color='tab:purple', linestyle=':', alpha=0.7)

    # Y-axis labels
    ref_lines = []
    if e_hf is not None:
        ref_lines.append((e_hf, 'red', f'HF: {e_hf:.4f}'))
    if e_ccsd is not None:
        ref_lines.append((e_ccsd, 'tab:green', f'CCSD: {e_ccsd:.4f}'))
    if e_random_best is not None:
        ref_lines.append((e_random_best, 'gray', f'Best Random-Only: {e_random_best:.4f}'))
    if e_best_hw is not None:
        ref_lines.append((e_best_hw, 'tab:orange', f'Best SQD: {e_best_hw:.4f}'))
    if e_best_sc is not None:
        ref_lines.append((e_best_sc, 'tab:purple', f'Best SC-SQD: {e_best_sc:.4f}'))
    add_ref_lines(ax, ref_lines)

    ax.set_xlabel('ADAPT iteration', fontsize=12)
    ax.set_ylabel('Energy (Ha)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.plot([], [], '*', color='gray', ms=14, alpha=0.5, label='In Progress')
    ax.legend(fontsize=9, loc='upper right')


SC_COLORS = ['tab:purple', 'tab:brown', 'darkviolet']
RANDOM_COLORS = ['dimgray', 'silver']


def make_fragment_plots(frag, e_hf, e_ccsd=None):
    """Generate standard plots for one fragment."""
    base = os.path.realpath(frag['base'])
    plots_dir = os.path.join(base, frag['plots_dir'])
    os.makedirs(plots_dir, exist_ok=True)

    # Load hardware variants
    hw_variants = {}
    for variant_name, variant_rel in frag['variants']:
        variant_dir = os.path.join(base, variant_rel)
        singletons, cumulatives = discover_and_load(variant_dir)
        if singletons or cumulatives:
            hw_variants[variant_name] = (singletons, cumulatives)

    if not hw_variants:
        print(f"  No results found for {frag['label']}, skipping plots")
        return

    # Load SC and random variants
    sc_loaded = []  # (label, runs_dict)
    for label, rel_path in frag.get('sc_variants', []):
        singleton, runs = discover_and_load_sc(os.path.join(base, rel_path))
        if runs:
            sc_loaded.append((label, runs))

    random_loaded = []
    random_singletons = []  # singleton results for "best random-only" reference line
    for label, rel_path in frag.get('random_variants', []):
        singleton, runs = discover_and_load_sc(os.path.join(base, rel_path))
        if singleton is not None:
            random_singletons.append(singleton)
        if runs:
            random_loaded.append((label, runs))

    # Best random singleton energy (SC/random only, no hardware data)
    e_random_best = None
    for result in random_singletons:
        if e_random_best is None or result['energy'] < e_random_best:
            e_random_best = result['energy']
    if e_random_best is not None:
        print(f"  Best random singleton: {e_random_best:.6f} Ha")

    # Best SC-SQD energy (from SC cumulatives only, not random)
    e_best_sc = None
    for _, runs in sc_loaded:
        for result in runs.values():
            if e_best_sc is None or result['energy'] < e_best_sc:
                e_best_sc = result['energy']
    if e_best_sc is not None:
        print(f"  Best SC-SQD: {e_best_sc:.6f} Ha")

    # Auto-detect all ADAPT iterations across hardware variants
    all_adapt_iters = set()
    for singletons, cumulatives in hw_variants.values():
        all_adapt_iters.update(singletons.keys())
        all_adapt_iters.update(cumulatives.keys())
    adapt_iters = sorted(all_adapt_iters)

    # --- Plot 1: Hardware-only energy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f'SQD Energy: {frag["label"]} ({frag["norb"]} orb, {frag["nelec"]} elec)'
    make_energy_plot(ax, hw_variants, [], adapt_iters, e_hf, e_ccsd,
                     e_random_best, None, title)
    fig.tight_layout()
    path = os.path.join(plots_dir, "singleton_vs_cumulative.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

    # --- Plot 2: Hardware-only iterations ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant_name, (singletons, cumulatives) in hw_variants.items():
        suffix = f' ({variant_name})' if len(hw_variants) > 1 else ''
        plot_series(ax, singletons, 'tab:blue', 'o', f'Singleton{suffix}',
                    adapt_iters, 'iters')
        plot_series(ax, cumulatives, 'tab:orange', 's', f'Cumulative{suffix}',
                    yidx='iters')
    ax.set_xlabel('ADAPT iteration', fontsize=12)
    ax.set_ylabel('SQD Iterations to Converge', fontsize=12)
    ax.set_title(f'SQD Convergence Speed: {frag["label"]}', fontsize=12)
    ax.plot([], [], '*', color='gray', ms=14, alpha=0.5, label='In Progress')
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(plots_dir, "singleton_vs_cumulative_iters.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

    # --- Plot 3: Determinants vs ADAPT ---
    has_dets = False
    for singletons, cumulatives in hw_variants.values():
        if any(v['determinants'] for v in singletons.values()):
            has_dets = True
        if any(v['determinants'] for v in cumulatives.values()):
            has_dets = True

    if has_dets:
        fig, ax = plt.subplots(figsize=(10, 6))
        for variant_name, (singletons, cumulatives) in hw_variants.items():
            suffix = f' ({variant_name})' if len(hw_variants) > 1 else ''
            sing_dets = {k: {'energy': v['determinants'], 'converged': v['converged']}
                         for k, v in singletons.items() if v['determinants']}
            cum_dets = {k: {'energy': v['determinants'], 'converged': v['converged']}
                        for k, v in cumulatives.items() if v['determinants']}
            if sing_dets:
                plot_series(ax, sing_dets, 'tab:blue', 'o', f'Singleton{suffix}',
                            adapt_iters, 'energy')
            if cum_dets:
                plot_series(ax, cum_dets, 'tab:orange', 's', f'Cumulative{suffix}',
                            yidx='energy')
        ax.set_xlabel('ADAPT iteration', fontsize=12)
        ax.set_ylabel('Determinants (final iteration)', fontsize=12)
        ax.set_title(f'CI Space Size: {frag["label"]}', fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(plots_dir, "determinants_vs_adapt.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    # --- SC/Random comparison plots (only if data exists) ---
    if sc_loaded:
        sc_series = [(label, SC_COLORS[i % len(SC_COLORS)], runs)
                     for i, (label, runs) in enumerate(sc_loaded)]
        fig, ax = plt.subplots(figsize=(10, 6))
        make_energy_plot(ax, hw_variants, sc_series, adapt_iters, e_hf, e_ccsd,
                         e_random_best, e_best_sc,
                         f'Hardware + SC: {frag["label"]}')
        fig.tight_layout()
        path = os.path.join(plots_dir, "hardware_plus_sc.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    if random_loaded:
        rand_series = [(label, RANDOM_COLORS[i % len(RANDOM_COLORS)], runs)
                       for i, (label, runs) in enumerate(random_loaded)]
        fig, ax = plt.subplots(figsize=(10, 6))
        make_energy_plot(ax, hw_variants, rand_series, adapt_iters, e_hf, e_ccsd,
                         e_random_best, None,
                         f'Hardware + Random: {frag["label"]}')
        fig.tight_layout()
        path = os.path.join(plots_dir, "hardware_plus_random.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    if sc_loaded and random_loaded:
        all_series = (
            [(label, SC_COLORS[i % len(SC_COLORS)], runs)
             for i, (label, runs) in enumerate(sc_loaded)]
            + [(label, RANDOM_COLORS[i % len(RANDOM_COLORS)], runs)
               for i, (label, runs) in enumerate(random_loaded)]
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        make_energy_plot(ax, hw_variants, all_series, adapt_iters, e_hf, e_ccsd,
                         e_random_best, e_best_sc,
                         f'Hardware + SC + Random: {frag["label"]}')
        fig.tight_layout()
        path = os.path.join(plots_dir, "hardware_plus_sc_plus_random.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)


def main():
    ccsd_data = load_ccsd_energies()
    if ccsd_data:
        print(f"Loaded CCSD energies for: {', '.join(ccsd_data.keys())}")

    for frag in FRAGMENTS:
        print(f"\n{'='*60}")
        print(f"Generating plots for {frag['label']}...")
        print(f"{'='*60}")

        # Compute HF energy
        e_hf = compute_hf_energy(frag['hamiltonian'])
        if e_hf is not None:
            print(f"  HF energy: {e_hf:.6f} Ha")

        # Look up CCSD energy
        e_ccsd = None
        frag_ccsd = ccsd_data.get(frag['fragment_name'])
        if frag_ccsd:
            e_ccsd = frag_ccsd['CCSD']
            print(f"  CCSD energy: {e_ccsd:.6f} Ha")
        else:
            print(f"  CCSD energy: not available for {frag['fragment_name']}")

        make_fragment_plots(frag, e_hf, e_ccsd)

    print("\nDone.")


if __name__ == "__main__":
    main()
