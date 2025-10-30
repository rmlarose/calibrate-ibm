"""Test 50k terms timing - serial v2 and parallel v2."""

import pickle
import time
import sys

sys.path.insert(0, '.')

print('Testing 50k terms with timing comparison')
print('='*60)

with open('tests/cached_50k_subset.pkl', 'rb') as f:
    P, nq = pickle.load(f)

from kcommute.sorted_insertion import get_si_sets_v2, get_terms_ordered_by_abscoeff
from kcommute.sorted_insertion_v2_parallel import get_si_sets_v2_parallel

terms = get_terms_ordered_by_abscoeff(P)
terms = [t for t in terms if () not in term.terms.keys()]
blocks = [list(range(nq))]

print(f'Testing {len(terms)} terms\n')

print('Running get_si_sets_v2...')
start = time.time()
g_serial = get_si_sets_v2(P, blocks, verbosity=0)
t_serial = time.time() - start
print(f'  Time: {t_serial:.3f}s, {len(g_serial)} groups')

print('\nRunning get_si_sets_v2_parallel...')
start = time.time()
g_parallel = get_si_sets_v2_parallel(terms, blocks, verbosity=0, num_workers=3, handshake_interval=75)
t_parallel = time.time() - start
speedup = t_serial / t_parallel
print(f'  Time: {t_parallel:.3f}s, {len(g_parallel)} groups, speedup: {speedup:.2f}x')

print(f'\n{"="*60}')
print(f'Summary:')
print(f'  get_si_sets_v2:          {t_serial:.3f}s, {len(g_serial)} groups (baseline)')
print(f'  get_si_sets_v2_parallel: {t_parallel:.3f}s, {len(g_parallel)} groups ({speedup:.2f}x faster)')
print(f'{"="*60}')

if len(g_serial) == len(g_parallel):
    print(f'\n✓ PASS: Both implementations produce {len(g_serial)} groups')
    sys.exit(0)
else:
    print(f'\n✗ FAIL: Implementations produce different numbers of groups')
    sys.exit(1)
