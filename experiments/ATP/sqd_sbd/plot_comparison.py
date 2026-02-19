"""Generate 4-trendline comparison plots: Sym vs No-Sym, Singleton vs Cumulative."""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})

# Reference energies
E_CCSD = -261.9430
E_HF = -261.6958

# --- Symmetrized results: (energy, iters, converged) ---
sing_sym = {
    1:  (-261.8131, 12,  True),
    2:  (-261.8020, 14,  True),
    3:  (-261.8358, 34,  True),
    4:  (-261.8498, 51,  True),
    5:  (-261.8632, 45,  True),
    10: (-261.8388, 275, True),
    20: (-261.8477, 353, True),
    25: (-261.8479, 376, True),
    30: (-261.8271, 372, False),
    40: (-261.8134, 220, True),
}

cumul_sym = {
    2:  (-261.8578, 5,   True),
    3:  (-261.8894, 38,  True),
    4:  (-261.8971, 49,  True),
    5:  (-261.9010, 78,  True),
    10: (-261.9124, 339, True),
    20: (-261.9171, 501, True),
    25: (-261.9155, 289, False),
    30: (-261.9157, 341, False),
    40: (-261.9154, 354, False),
}

# --- No-symmetrization results (read from energy files) ---
nosym_dir = "./nosym/results"

# Hardcoded convergence status (from SLURM log analysis).
# SLURM array indices that converged: {0,1,2,3,4,5,7,10,11,12,13}
# Singletons: 0->1, 1->2, 2->3, 3->4, 4->5, 5->10, 6->20, 7->25, 8->30, 9->40
# Cumulatives: 10->[1,2], 11->[1,2,3], 12->[1,..,4], 13->[1,..,5], 14->[1,..,10],
#              15->[1,..,20], 16->[1,..,25], 17->[1,..,30], 18->[1,..,40]
nosym_converged_indices = {0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13}

# Array index -> ADAPT mapping for singletons
sing_idx_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 10, 6: 20, 7: 25, 8: 30, 9: 40}
# Array index -> cumulative x-axis (last ADAPT in set)
cumul_idx_map = {10: 2, 11: 3, 12: 4, 13: 5, 14: 10, 15: 20, 16: 25, 17: 30, 18: 40}

sing_nosym = {}
for idx, adapt in sing_idx_map.items():
    efile = os.path.join(nosym_dir, f"singleton_{adapt}", f"sqd_energies_{adapt}.txt")
    if os.path.exists(efile):
        energies = np.loadtxt(efile)
        if energies.ndim == 0:
            energies = np.array([float(energies)])
        sing_nosym[adapt] = (float(energies[-1]), len(energies), idx in nosym_converged_indices)

cumul_nosym = {}
cumul_sets = [
    [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 5, 10, 20],
    [1, 2, 3, 4, 5, 10, 20, 25], [1, 2, 3, 4, 5, 10, 20, 25, 30],
    [1, 2, 3, 4, 5, 10, 20, 25, 30, 40],
]
for i, cset in enumerate(cumul_sets):
    key = "_".join(map(str, cset))
    x = cset[-1]
    idx = 10 + i  # array indices 10-18
    efile = os.path.join(nosym_dir, f"cumulative_{key}", f"sqd_energies_{key}.txt")
    if os.path.exists(efile):
        energies = np.loadtxt(efile)
        if energies.ndim == 0:
            energies = np.array([float(energies)])
        cumul_nosym[x] = (float(energies[-1]), len(energies), idx in nosym_converged_indices)


def plot_series(ax, data, yidx, style, color, ms, label, alpha=1.0, mfc=None):
    """Plot a data series with stars for in-progress points."""
    xs = sorted(data.keys())
    ys = [data[x][yidx] for x in xs]

    conv_x = [x for x in xs if data[x][2]]
    conv_y = [data[x][yidx] for x in conv_x]
    prog_x = [x for x in xs if not data[x][2]]
    prog_y = [data[x][yidx] for x in prog_x]

    # Connecting line through all points
    ax.plot(xs, ys, style[0], color=color, alpha=alpha, zorder=1)
    # Converged markers
    if mfc is None:
        mfc_val = color
    else:
        mfc_val = mfc
    ax.plot(conv_x, conv_y, style[1], color=color, ms=ms, mfc=mfc_val,
            alpha=alpha, label=label, zorder=2)
    # In-progress stars
    if prog_x:
        ax.plot(prog_x, prog_y, '*', color='gray', ms=14, mec='gray', zorder=3)

    return prog_x  # return so we know if any were plotted


def make_plot(yidx, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    any_prog = []
    any_prog += plot_series(ax, sing_sym, yidx, ('-', 'o'), 'tab:blue', 8,
                            'Singleton (Sym)')
    any_prog += plot_series(ax, cumul_sym, yidx, ('-', 's'), 'tab:orange', 8,
                            'Cumulative (Sym)')
    any_prog += plot_series(ax, sing_nosym, yidx, ('--', 'o'), 'tab:blue', 6,
                            'Singleton (No Sym)', alpha=0.6, mfc='none')
    any_prog += plot_series(ax, cumul_nosym, yidx, ('--', 's'), 'tab:orange', 6,
                            'Cumulative (No Sym)', alpha=0.6, mfc='none')

    # Single "In progress" legend entry
    if any_prog:
        ax.plot([], [], '*', color='gray', ms=14, mec='gray', label='In progress')

    if yidx == 0:
        ax.axhline(y=E_CCSD, color='green', linestyle='--', label='CCSD')
        ax.axhline(y=E_HF, color='red', linestyle='--', label='HF')

        all_e = ([v[0] for v in sing_sym.values()] +
                 [v[0] for v in cumul_sym.values()] +
                 [v[0] for v in sing_nosym.values()] +
                 [v[0] for v in cumul_nosym.values()])
        E_best = min(all_e)
        ax.axhline(y=E_best, color='purple', linestyle=':', alpha=0.7, label='Best SQD')

        ref_vals = [E_CCSD, E_HF, E_best]
        default_ticks = [t for t in ax.get_yticks()
                         if all(abs(t - r) > 0.02 for r in ref_vals)]
        yticks = list(default_ticks) + ref_vals
        ytick_labels = [f'{t:.2f}' for t in default_ticks] + \
                       [f'{v:.4f}' for v in ref_vals]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)

    ax.set_xlabel('ADAPT iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved {filename}")


make_plot(0, "Energy (Ha)", "./plots/sym_vs_nosym_energy.png")
make_plot(1, "Iterations", "./plots/sym_vs_nosym_iters.png")
