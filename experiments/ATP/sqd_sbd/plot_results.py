"""Generate singleton vs cumulative energy and iteration plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})

# ADAPT iteration x-axis values
adapt_iters = [1, 2, 3, 4, 5, 10, 20, 25, 30, 40]

# Singleton no-S² results: (energy, iters, converged)
singletons = {
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

# Cumulative no-S² results: x-axis = last ADAPT iter in set
cumulatives = {
    2:  (-261.8578, 5,   True),   # [1,2]
    3:  (-261.8894, 38,  True),   # [1,2,3]
    4:  (-261.8971, 49,  True),   # [1,2,3,4]
    5:  (-261.9010, 78,  True),   # [1,2,3,4,5]
    10: (-261.9124, 339, True),   # [1,...,10]
    20: (-261.9171, 501, True),   # [1,...,20]
    25: (-261.9155, 289, False),  # [1,...,25]
    30: (-261.9157, 341, False),  # [1,...,30]
    40: (-261.9154, 354, False),  # [1,...,40]
}

# Reference energies
E_CCSD = -261.9430
E_HF = -261.6958


def make_plot(ykey, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    yidx = 0 if ykey == "energy" else 1

    # Singleton converged
    s_conv_x = [x for x in adapt_iters if x in singletons and singletons[x][2]]
    s_conv_y = [singletons[x][yidx] for x in s_conv_x]
    # Singleton in-progress
    s_prog_x = [x for x in adapt_iters if x in singletons and not singletons[x][2]]
    s_prog_y = [singletons[x][yidx] for x in s_prog_x]

    # Cumulative converged
    c_x_all = sorted(cumulatives.keys())
    c_conv_x = [x for x in c_x_all if cumulatives[x][2]]
    c_conv_y = [cumulatives[x][yidx] for x in c_conv_x]
    # Cumulative in-progress
    c_prog_x = [x for x in c_x_all if not cumulatives[x][2]]
    c_prog_y = [cumulatives[x][yidx] for x in c_prog_x]

    # Draw connecting lines through ALL points (converged + in-progress)
    s_all_x = sorted(singletons.keys())
    s_all_y = [singletons[x][yidx] for x in s_all_x]
    c_all_x = sorted(cumulatives.keys())
    c_all_y = [cumulatives[x][yidx] for x in c_all_x]

    ax.plot(s_all_x, s_all_y, '-', color='tab:blue', zorder=1)
    ax.plot(c_all_x, c_all_y, '-', color='tab:orange', zorder=1)

    # Converged markers
    ax.plot(s_conv_x, s_conv_y, 'o', color='tab:blue', ms=8, mec='tab:blue',
            label='Singleton', zorder=2)
    ax.plot(c_conv_x, c_conv_y, 's', color='tab:orange', ms=8, mec='tab:orange',
            label='Cumulative', zorder=2)

    # In-progress markers (gray stars, single legend entry)
    first_prog = True
    for px, py in [(s_prog_x, s_prog_y), (c_prog_x, c_prog_y)]:
        if px:
            ax.plot(px, py, '*', color='gray', ms=14, mec='gray',
                    label='In progress' if first_prog else None, zorder=3)
            first_prog = False

    if ykey == "energy":
        ax.axhline(y=E_CCSD, color='green', linestyle='--', label='CCSD')
        ax.axhline(y=E_HF, color='red', linestyle='--', label='HF')

        # Best energy across all data
        all_energies = [v[0] for v in singletons.values()] + [v[0] for v in cumulatives.values()]
        E_best = min(all_energies)
        ax.axhline(y=E_best, color='purple', linestyle=':', alpha=0.7, label='Best SQD')

        # Add labeled ticks on y-axis for reference energies,
        # removing any default ticks that would overlap
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


make_plot("energy", "Energy (Ha)", "./plots/singleton_vs_cumulative.png")
make_plot("iters", "Iterations", "./plots/singleton_vs_cumulative_iters.png")
