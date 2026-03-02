"""Test vectorized recover_configurations against the original."""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from qiskit_addon_sqd.configuration_recovery import recover_configurations
from run_sqd_sbd import recover_configurations_fast

# Test parameters matching f2 (44 orbitals, 22 alpha, 22 beta)
norb = 44
n_alpha = 22
n_beta = 22
num_bits = 2 * norb
N = 14000

rng = np.random.default_rng(42)

# Generate random bitstrings with roughly correct hamming weight
bs_matrix = np.zeros((N, num_bits), dtype=bool)
for i in range(N):
    n_left = min(max(n_beta + rng.integers(-3, 4), 0), norb)
    n_right = min(max(n_alpha + rng.integers(-3, 4), 0), norb)
    left_ones = rng.choice(norb, size=n_left, replace=False)
    right_ones = rng.choice(norb, size=n_right, replace=False)
    bs_matrix[i, left_ones] = True
    bs_matrix[i, norb + right_ones] = True

probs = rng.random(N)
probs /= probs.sum()

# Random occupancies
occ_alpha = rng.random(norb) * 0.8 + 0.1
occ_beta = rng.random(norb) * 0.8 + 0.1
avg_occ = (occ_alpha, occ_beta)

# Time the original
rng1 = np.random.default_rng(123)
t0 = time.time()
mat1, freq1 = recover_configurations(bs_matrix, probs, avg_occ, n_alpha, n_beta, rand_seed=rng1)
t_old = time.time() - t0

# Time the fast version
rng2 = np.random.default_rng(123)
t0 = time.time()
mat2, freq2 = recover_configurations_fast(bs_matrix, probs, avg_occ, n_alpha, n_beta, rand_seed=rng2)
t_new = time.time() - t0

print(f'Original: {t_old:.3f}s, {mat1.shape[0]} unique bitstrings')
print(f'Fast:     {t_new:.3f}s, {mat2.shape[0]} unique bitstrings')
print(f'Speedup:  {t_old/t_new:.1f}x')

# Verify correctness: all output bitstrings have correct hamming weight
left_hw1 = mat1[:, :norb].sum(axis=1)
right_hw1 = mat1[:, norb:].sum(axis=1)
left_hw2 = mat2[:, :norb].sum(axis=1)
right_hw2 = mat2[:, norb:].sum(axis=1)

print(f'\nOriginal hamming: left={np.unique(left_hw1)}, right={np.unique(right_hw1)}')
print(f'Fast hamming:     left={np.unique(left_hw2)}, right={np.unique(right_hw2)}')
print(f'Probabilities sum: old={freq1.sum():.6f}, new={freq2.sum():.6f}')
print(f'All probs non-negative: old={np.all(freq1 >= 0)}, new={np.all(freq2 >= 0)}')

# Also test with metaphosphate-sized data (22 orb, 46k bitstrings)
print('\n--- Metaphosphate-sized test (22 orb, 46k bitstrings) ---')
norb2 = 22
n_alpha2 = 16
n_beta2 = 16
num_bits2 = 2 * norb2
N2 = 46000

rng = np.random.default_rng(99)
bs2 = np.zeros((N2, num_bits2), dtype=bool)
for i in range(N2):
    n_left = min(max(n_beta2 + rng.integers(-3, 4), 0), norb2)
    n_right = min(max(n_alpha2 + rng.integers(-3, 4), 0), norb2)
    bs2[i, rng.choice(norb2, size=n_left, replace=False)] = True
    bs2[i, norb2 + rng.choice(norb2, size=n_right, replace=False)] = True

probs2 = rng.random(N2)
probs2 /= probs2.sum()
occ2 = (rng.random(norb2) * 0.8 + 0.1, rng.random(norb2) * 0.8 + 0.1)

rng1 = np.random.default_rng(456)
t0 = time.time()
m1, f1 = recover_configurations(bs2, probs2, occ2, n_alpha2, n_beta2, rand_seed=rng1)
t_old2 = time.time() - t0

rng2 = np.random.default_rng(456)
t0 = time.time()
m2, f2 = recover_configurations_fast(bs2, probs2, occ2, n_alpha2, n_beta2, rand_seed=rng2)
t_new2 = time.time() - t0

print(f'Original: {t_old2:.3f}s, {m1.shape[0]} unique')
print(f'Fast:     {t_new2:.3f}s, {m2.shape[0]} unique')
print(f'Speedup:  {t_old2/t_new2:.1f}x')

left_hw = m2[:, :norb2].sum(axis=1)
right_hw = m2[:, norb2:].sum(axis=1)
print(f'Fast hamming: left={np.unique(left_hw)}, right={np.unique(right_hw)}')
