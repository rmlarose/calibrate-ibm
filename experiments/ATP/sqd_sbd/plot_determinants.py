"""Plot #determinants vs ADAPT iteration for all 4 trendlines."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})

# --- Symmetrized determinant counts (final iteration, max of batch 0/1) ---
sing_sym_dets = {
    1: 27889, 2: 27889, 3: 32400, 4: 28224, 5: 26896,
    10: 23104, 20: 30976, 25: 37636, 30: 55696, 40: 49284,
}

cumul_sym_dets = {
    2: 68644, 3: 44944, 4: 37636, 5: 35344, 10: 33856, 20: 33856, 25: 38025, 30: 53824, 40: 62001,
}

# --- No-symmetrization determinant counts ---
sing_nosym_dets = {
    1: 10998, 2: 12519, 3: 14508, 4: 12726, 5: 12444,
    10: 10112, 20: 13334, 25: 14040, 30: 26196, 40: 23760,
}

cumul_nosym_dets = {
    2: 31140, 3: 19599, 4: 14224, 5: 15390, 10: 12220,
    20: 16368, 25: 15609, 30: 24940, 40: 24624,
}

# --- Convergence status (for star markers) ---
# Sym
sing_sym_conv = {1: True, 2: True, 3: True, 4: True, 5: True,
                 10: True, 20: True, 25: True, 30: False, 40: True}
cumul_sym_conv = {2: True, 3: True, 4: True, 5: True, 10: True,
                  20: True, 25: False, 30: False, 40: False}

# Nosym: hardcoded from SLURM log analysis.
# Converged SLURM array indices: {0,1,2,3,4,5,7,10,11,12,13}
sing_idx_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 10, 6: 20, 7: 25, 8: 30, 9: 40}
cumul_idx_map = {10: 2, 11: 3, 12: 4, 13: 5, 14: 10, 15: 20, 16: 25, 17: 30, 18: 40}
nosym_converged_indices = {0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13}

sing_nosym_conv = {adapt: (idx in nosym_converged_indices)
                   for idx, adapt in sing_idx_map.items()}
cumul_nosym_conv = {adapt: (idx in nosym_converged_indices)
                    for idx, adapt in cumul_idx_map.items()}


def plot_series(ax, dets, conv, style, color, ms, label, alpha=1.0, mfc=None):
    xs = sorted(dets.keys())
    ys = [dets[x] for x in xs]

    conv_x = [x for x in xs if conv.get(x, False)]
    conv_y = [dets[x] for x in conv_x]
    prog_x = [x for x in xs if not conv.get(x, False)]
    prog_y = [dets[x] for x in prog_x]

    ax.plot(xs, ys, style[0], color=color, alpha=alpha, zorder=1)
    ax.plot(conv_x, conv_y, style[1], color=color, ms=ms,
            mfc=mfc if mfc else color, alpha=alpha, label=label, zorder=2)
    if prog_x:
        ax.plot(prog_x, prog_y, '*', color='gray', ms=14, mec='gray', zorder=3)
    return prog_x


fig, ax = plt.subplots(figsize=(10, 6))

any_prog = []
any_prog += plot_series(ax, sing_sym_dets, sing_sym_conv,
                        ('-', 'o'), 'tab:blue', 8, 'Singleton (Sym)')
any_prog += plot_series(ax, cumul_sym_dets, cumul_sym_conv,
                        ('-', 's'), 'tab:orange', 8, 'Cumulative (Sym)')
any_prog += plot_series(ax, sing_nosym_dets, sing_nosym_conv,
                        ('--', 'o'), 'tab:blue', 6, 'Singleton (No Sym)',
                        alpha=0.6, mfc='none')
any_prog += plot_series(ax, cumul_nosym_dets, cumul_nosym_conv,
                        ('--', 's'), 'tab:orange', 6, 'Cumulative (No Sym)',
                        alpha=0.6, mfc='none')

if any_prog:
    ax.plot([], [], '*', color='gray', ms=14, mec='gray', label='In progress')

ax.set_xlabel('ADAPT iteration', fontsize=12)
ax.set_ylabel('Determinants', fontsize=12)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("./plots/sym_vs_nosym_dets.png", dpi=150)
print("Saved ./plots/sym_vs_nosym_dets.png")
