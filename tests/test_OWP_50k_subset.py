"""Test 50k terms correctness - serial v2 and parallel v2."""

import pickle
import sys

sys.path.insert(0, '.')

print('Testing 50k terms correctness')
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
g_serial = get_si_sets_v2(P, blocks, verbosity=0)

print('Running get_si_sets_v2_parallel...')
g_parallel = get_si_sets_v2_parallel(terms, blocks, verbosity=0, num_workers=3, handshake_interval=75)

print(f'\nResults:')
print(f'  get_si_sets_v2:          {len(g_serial)} groups')
print(f'  get_si_sets_v2_parallel: {len(g_parallel)} groups')

if len(g_serial) == len(g_parallel):
    print(f'\n✓ PASS: Both implementations produce {len(g_serial)} groups')
    sys.exit(0)
else:
    print(f'\n✗ FAIL: Implementations produce different numbers of groups')
    sys.exit(1)
