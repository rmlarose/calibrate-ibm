"""Diagnose why v5_parallel_v2 isn't scaling on larger systems.

Focus on P_3 (20k terms) to understand bottlenecks.
"""

import os
import sys
import time
import numpy as np
from openfermion import QubitOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Hamiltonians.load_owp import load_owp_hamiltonian


def create_random_subset(hamiltonian, n_terms, seed=42):
    """Create a random subset of terms."""
    np.random.seed(seed)
    all_terms = list(hamiltonian.terms.items())

    if n_terms >= len(all_terms):
        return hamiltonian

    indices = np.random.choice(len(all_terms), size=n_terms, replace=False)
    sampled_terms = [all_terms[i] for i in indices]

    subset = QubitOperator()
    for pauli_string, coeff in sampled_terms:
        subset += QubitOperator(pauli_string, coeff)

    return subset


def profile_v5_serial(hamiltonian, blocks, verbosity=1):
    """Profile v5 serial to get baseline."""
    from kcommute.sorted_insertion import get_si_sets_v5

    print("="*80)
    print("PROFILING v5 (SERIAL)")
    print("="*80)

    start = time.time()
    groups = get_si_sets_v5(hamiltonian, blocks, verbosity=verbosity)
    elapsed = time.time() - start

    print(f"\nv5 serial: {elapsed:.3f}s ({len(groups)} groups)")
    return elapsed, len(groups)


def profile_v5_parallel_v2(hamiltonian, blocks, num_workers, verbosity=1):
    """Profile v5_parallel_v2 with instrumentation."""
    import multiprocessing as mp
    import numpy as np
    import kcommute.commute
    from kcommute.sorted_insertion import get_terms_ordered_by_abscoeff

    print("="*80)
    print(f"PROFILING v5_parallel_v2 ({num_workers} workers)")
    print("="*80)

    # Same setup as v5_parallel_v2
    terms_ord = get_terms_ordered_by_abscoeff(hamiltonian)
    terms_ord = [term for term in terms_ord if () not in term.terms.keys()]

    terms_symplectic_list = [kcommute.commute.pauli_to_symplectic(term) for term in terms_ord]
    x_bits_array = np.array([x for x, z in terms_symplectic_list], dtype=np.int64)
    z_bits_array = np.array([z for x, z in terms_symplectic_list], dtype=np.int64)
    support_array = x_bits_array | z_bits_array

    # Pre-compute block masks
    block_masks = []
    for block in blocks:
        block_mask = 0
        for qubit_idx in block:
            block_mask |= (1 << qubit_idx)
        block_masks.append(block_mask)

    nqubits = max(max(block) for block in blocks) if blocks else 0
    all_bits_valid = (1 << (nqubits + 1)) - 1

    is_single_block = len(blocks) == 1
    if is_single_block:
        single_block_mask = block_masks[0]

    # Bootstrap phase
    bootstrap_size = min(100, len(terms_ord))
    commuting_sets = []

    print(f"Bootstrap: {bootstrap_size} terms")
    bootstrap_start = time.time()

    for i in range(bootstrap_size):
        pstring = terms_ord[i]
        x_bits = int(x_bits_array[i])
        z_bits = int(z_bits_array[i])
        support_mask = int(support_array[i])

        found_commuting_set = False

        for group_data in commuting_sets:
            terms_list, group_x, group_z, valid_bits = group_data
            anticommute_mask = (x_bits & group_z) ^ (z_bits & group_x)
            support_in_invalid_region = support_mask & ~valid_bits

            if support_in_invalid_region == 0 and (anticommute_mask & valid_bits) == 0:
                terms_list.append((pstring, (x_bits, z_bits), support_mask))
                group_data[1] = group_x | x_bits
                group_data[2] = group_z | z_bits
                found_commuting_set = True
                break

            all_strings_in_commset_commute = True
            for pstring2, (x2, z2), support_mask2 in terms_list:
                anticommute_mask_full = (x_bits & z2) ^ (z_bits & x2)
                if anticommute_mask_full == 0:
                    continue

                if is_single_block:
                    inner_product = anticommute_mask_full & single_block_mask
                    if bin(inner_product).count('1') % 2 != 0:
                        all_strings_in_commset_commute = False
                        break
                else:
                    commutes_in_all_blocks = True
                    for block_mask in block_masks:
                        inner_product = anticommute_mask_full & block_mask
                        if bin(inner_product).count('1') % 2 != 0:
                            commutes_in_all_blocks = False
                            break
                    if not commutes_in_all_blocks:
                        all_strings_in_commset_commute = False
                        break

            if all_strings_in_commset_commute:
                terms_list.append((pstring, (x_bits, z_bits), support_mask))
                group_data[1] = group_x | x_bits
                group_data[2] = group_z | z_bits
                group_data[3] = valid_bits & ~anticommute_mask
                found_commuting_set = True
                break

        if not found_commuting_set:
            commuting_sets.append([[(pstring, (x_bits, z_bits), support_mask)], x_bits, z_bits, all_bits_valid])

    bootstrap_time = time.time() - bootstrap_start
    print(f"Bootstrap time: {bootstrap_time:.3f}s ({len(commuting_sets)} groups)")

    # Main loop with instrumentation
    main_start = time.time()

    handshake_count = 0
    handshake_time = 0.0
    exclusion_receive_count = 0
    exclusion_receive_time = 0.0
    total_exclusions_received = 0
    grouping_time = 0.0

    exclusions = {}
    handshake_interval = 100
    last_handshake = bootstrap_size

    for i in range(bootstrap_size, min(bootstrap_size + 5000, len(terms_ord))):  # Limit to 5k terms for quick diagnosis
        # Handshake timing
        if i - last_handshake >= handshake_interval:
            hs_start = time.time()
            handshake_count += 1

            # Simulate handshake (we're not actually running workers here, just measuring overhead)
            # In real version: send position, receive exclusions

            handshake_time += time.time() - hs_start
            last_handshake = i

        # Grouping timing
        group_start = time.time()

        pstring = terms_ord[i]
        x_bits = int(x_bits_array[i])
        z_bits = int(z_bits_array[i])
        support_mask = int(support_array[i])

        excluded_groups = exclusions.get(i, set())
        found_commuting_set = False

        for group_id, group_data in enumerate(commuting_sets):
            if group_id in excluded_groups:
                continue

            terms_list, group_x, group_z, valid_bits = group_data
            anticommute_mask = (x_bits & group_z) ^ (z_bits & group_x)
            support_in_invalid_region = support_mask & ~valid_bits

            if support_in_invalid_region == 0 and (anticommute_mask & valid_bits) == 0:
                terms_list.append((pstring, (x_bits, z_bits), support_mask))
                group_data[1] = group_x | x_bits
                group_data[2] = group_z | z_bits
                found_commuting_set = True
                break

            all_strings_in_commset_commute = True
            for pstring2, (x2, z2), support_mask2 in terms_list:
                anticommute_mask_full = (x_bits & z2) ^ (z_bits & x2)
                if anticommute_mask_full == 0:
                    continue

                if is_single_block:
                    inner_product = anticommute_mask_full & single_block_mask
                    if bin(inner_product).count('1') % 2 != 0:
                        all_strings_in_commset_commute = False
                        break
                else:
                    commutes_in_all_blocks = True
                    for block_mask in block_masks:
                        inner_product = anticommute_mask_full & block_mask
                        if bin(inner_product).count('1') % 2 != 0:
                            commutes_in_all_blocks = False
                            break
                    if not commutes_in_all_blocks:
                        all_strings_in_commset_commute = False
                        break

            if all_strings_in_commset_commute:
                terms_list.append((pstring, (x_bits, z_bits), support_mask))
                group_data[1] = group_x | x_bits
                group_data[2] = group_z | z_bits
                group_data[3] = valid_bits & ~anticommute_mask
                found_commuting_set = True
                break

        if not found_commuting_set:
            commuting_sets.append([[(pstring, (x_bits, z_bits), support_mask)], x_bits, z_bits, all_bits_valid])

        grouping_time += time.time() - group_start

    main_time = time.time() - main_start

    print(f"\nMain loop time breakdown:")
    print(f"  Total time: {main_time:.3f}s")
    print(f"  Grouping computation: {grouping_time:.3f}s ({grouping_time/main_time*100:.1f}%)")
    print(f"  Handshake overhead: {handshake_time:.3f}s ({handshake_time/main_time*100:.1f}%)")
    print(f"  Handshakes: {handshake_count}")
    print(f"  Groups created: {len(commuting_sets)}")

    total_time = bootstrap_time + main_time
    return total_time, len(commuting_sets)


if __name__ == "__main__":
    print("="*80)
    print("PARALLEL BOTTLENECK DIAGNOSIS")
    print("="*80)

    # Load and create P_3 subset
    print("\nLoading OWP Hamiltonian...")
    npz_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'owp_reactant.npz')
    hamiltonian_full, nqubits, nterms_full = load_owp_hamiltonian(npz_path)

    print("\nCreating P_3 (20k terms) subset...")
    P_3_full = create_random_subset(hamiltonian_full, 20000, seed=42)
    P_2 = create_random_subset(P_3_full, 10000, seed=43)
    P_3 = create_random_subset(P_2, 5000, seed=44)  # Use 5k for faster diagnosis

    print(f"Using subset: {len(P_3.terms)} terms")

    blocks_kN = [list(range(nqubits))]

    # Profile serial
    time_v5, groups_v5 = profile_v5_serial(P_3, blocks_kN, verbosity=0)

    # Profile parallel (instrumented)
    print()
    time_parallel, groups_parallel = profile_v5_parallel_v2(P_3, blocks_kN, num_workers=2, verbosity=0)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nv5 serial: {time_v5:.3f}s")
    print(f"v5_parallel_v2 (instrumented): {time_parallel:.3f}s")
    print(f"Overhead: {(time_parallel - time_v5):.3f}s ({(time_parallel/time_v5 - 1)*100:.1f}%)")
