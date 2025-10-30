"""Test 50k terms comparing serial v2 and parallel v2_parallel with interval=75."""

import pickle
import time
import sys

sys.path.insert(0, '.')

print('Testing interval=75 on 50k terms')
print('='*60)

with open('tests/cached_50k_subset.pkl', 'rb') as f:
    P, nq = pickle.load(f)

from kcommute.sorted_insertion import get_si_sets_v2, get_terms_ordered_by_abscoeff
from kcommute.sorted_insertion_v2_parallel import get_si_sets_v2_parallel

terms = get_terms_ordered_by_abscoeff(P)
terms = [t for t in terms if () not in t.terms.keys()]
blocks = [list(range(nq))]

print(f'Testing {len(terms)} terms\n')

# v2 baseline (serial)
print('v2 (serial):    ', end=' ', flush=True)
start = time.time()
g_serial = get_si_sets_v2(P, blocks, verbosity=0)
t_serial = time.time() - start
print(f'{t_serial:.3f}s, {len(g_serial)} groups')

# v2_parallel with interval=75
print('v2_parallel:    ', end=' ', flush=True)
start = time.time()
g_parallel = get_si_sets_v2_parallel(terms, blocks, verbosity=0, num_workers=3, handshake_interval=75)
t_parallel = time.time() - start
speedup = t_serial/t_parallel
print(f'{t_parallel:.3f}s, {len(g_parallel)} groups, speedup={speedup:.2f}x')

print(f'\nCorrectness: {"PASS" if len(g_serial)==len(g_parallel) else "FAIL"}')
print('Process terminated cleanly!')
