import multiprocessing as mp
import numpy as np
import time


def _popcount_parity(n):
    try:
        return n.bit_count() & 1
    except AttributeError:
        count = 0
        while n:
            n &= n - 1
            count += 1
        return count & 1


def _worker(worker_id, worker_start, worker_end, cache_path, blocks,
            broadcast_queue, exclusion_queue, stop_event,
            work_request_queue, work_response_queue):
    cache_data = np.load(cache_path)
    x_bits_array = cache_data['x_bits']
    z_bits_array = cache_data['z_bits']

    block_masks = []
    for block in blocks:
        mask = 0
        for idx in block:
            mask |= (1 << idx)
        block_masks.append(mask)

    groups = []
    parent_position = 0
    last_num_groups = 0
    exclusions = {}
    exclusions_sent = {}
    current_position = worker_end - 1

    is_single_block = len(blocks) == 1
    if is_single_block:
        single_block_mask = block_masks[0]

    while not stop_event.is_set():
        try:
            msg_type, new_parent_pos, incremental_updates = broadcast_queue.get_nowait()
            if msg_type == 'stop':
                break
            elif msg_type == 'update':
                parent_position = new_parent_pos
                old_num_groups = len(groups)

                for group_id, gx, gz, vb, new_terms in incremental_updates:
                    while len(groups) <= group_id:
                        groups.append((0, 0, (1 << 64) - 1, []))

                    old_gx, old_gz, old_vb, term_list = groups[group_id]
                    term_list.extend(new_terms)
                    groups[group_id] = (gx, gz, vb, term_list)

                if len(groups) > last_num_groups and current_position < effective_lower_bound:
                    current_position = worker_end - 1
                last_num_groups = len(groups)

                if parent_position >= worker_start - 200:
                    ready_exclusions = {}
                    for idx, excl_set in exclusions.items():
                        if idx >= parent_position + 100 and len(excl_set) > 0:
                            already_sent = exclusions_sent.get(idx, set())
                            new_excl = excl_set - already_sent
                            if len(new_excl) > 0:
                                ready_exclusions[idx] = new_excl
                                exclusions_sent[idx] = already_sent | new_excl

                    if len(ready_exclusions) > 0:
                        exclusion_queue.put(ready_exclusions)
        except:
            pass

        effective_lower_bound = max(worker_start, parent_position + 200)
        if current_position >= effective_lower_bound and len(groups) > 0:
            term_idx = current_position
            x = int(x_bits_array[term_idx])
            z = int(z_bits_array[term_idx])
            support = x | z

            if term_idx not in exclusions:
                exclusions[term_idx] = set()

            for group_id, (gx, gz, valid, term_list) in enumerate(groups):
                if group_id in exclusions[term_idx]:
                    continue

                anticommute = (x & gz) ^ (z & gx)
                invalid_support = support & ~valid

                if invalid_support == 0 and (anticommute & valid) == 0:
                    continue

                found_anticommute = False
                for x2, z2 in term_list:
                    anticommute_full = (x & z2) ^ (z & x2)
                    if anticommute_full == 0:
                        continue

                    if is_single_block:
                        if _popcount_parity(anticommute_full & single_block_mask):
                            found_anticommute = True
                            break
                    else:
                        for mask in block_masks:
                            if _popcount_parity(anticommute_full & mask):
                                found_anticommute = True
                                break
                        if found_anticommute:
                            break

                if found_anticommute:
                    exclusions[term_idx].add(group_id)

            current_position -= 1

        elif current_position < effective_lower_bound:
            if parent_position >= worker_end - 200:
                final_exclusions = {}
                for idx, excl_set in exclusions.items():
                    if len(excl_set) > 0:
                        already_sent = exclusions_sent.get(idx, set())
                        new_excl = excl_set - already_sent
                        if len(new_excl) > 0:
                            final_exclusions[idx] = new_excl

                if len(final_exclusions) > 0:
                    exclusion_queue.put(final_exclusions)

                work_request_queue.put(worker_id)
                try:
                    new_start, new_end = work_response_queue.get(timeout=0.5)
                    if new_start is None:
                        break

                    all_remaining = {}
                    for idx, excl_set in exclusions.items():
                        if idx < new_start or idx >= new_end:
                            if len(excl_set) > 0:
                                already_sent = exclusions_sent.get(idx, set())
                                new_excl = excl_set - already_sent
                                if len(new_excl) > 0:
                                    all_remaining[idx] = new_excl

                    if len(all_remaining) > 0:
                        exclusion_queue.put(all_remaining)

                    exclusions = {}
                    exclusions_sent = {}
                    worker_start = new_start
                    worker_end = new_end
                    current_position = worker_end - 1
                except:
                    if stop_event.is_set():
                        break
                    time.sleep(0.01)
                    continue
            else:
                time.sleep(0.001)
        else:
            time.sleep(0.001)


def get_si_sets_v2_parallel(terms_ord, blocks, verbosity: int = 0, num_workers: int = 2,
                            handshake_interval: int = 100, chunk_divisor: int = 1):
    import kcommute
    import os
    import hashlib

    cache_dir = os.path.expanduser("~/.kcommute_cache")
    os.makedirs(cache_dir, exist_ok=True)

    terms_str = str([(term.terms, list(term.terms.values())[0] if term.terms else 0) for term in terms_ord])
    cache_key = hashlib.md5(terms_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"symplectic_{cache_key}.npz")

    cache_start = time.time()
    if os.path.exists(cache_path):
        if verbosity > 0:
            print(f"Loading cached arrays from {cache_path}")
        cache_data = np.load(cache_path)
        x_bits_array = cache_data['x_bits']
        z_bits_array = cache_data['z_bits']
        support_array = cache_data['support']
        if verbosity > 0:
            print(f"  Cache load time: {time.time() - cache_start:.3f}s")
    else:
        if verbosity > 0:
            print(f"Computing and caching arrays to {cache_path}")
        terms_symplectic = [kcommute.commute.pauli_to_symplectic(term) for term in terms_ord]
        x_bits_array = np.array([x for x, z in terms_symplectic], dtype=np.uint64)
        z_bits_array = np.array([z for x, z in terms_symplectic], dtype=np.uint64)
        support_array = np.array([x | z for x, z in terms_symplectic], dtype=np.uint64)
        np.savez(cache_path, x_bits=x_bits_array, z_bits=z_bits_array, support=support_array)
        if verbosity > 0:
            print(f"  Cache creation time: {time.time() - cache_start:.3f}s")

    block_masks = []
    for block in blocks:
        mask = 0
        for idx in block:
            mask |= (1 << idx)
        block_masks.append(mask)

    N = len(terms_ord)
    bootstrap_size = min(20, N)

    max_qubit = max(max(block) for block in blocks) if blocks else 0
    all_bits_valid = (1 << (max_qubit + 1)) - 1

    commuting_sets = []
    is_single_block = len(blocks) == 1
    single_block_mask = block_masks[0] if is_single_block else None

    for i in range(bootstrap_size):
        pstring = terms_ord[i]
        x = int(x_bits_array[i])
        z = int(z_bits_array[i])
        support = int(support_array[i])

        found = False
        for group_data in commuting_sets:
            terms_list, gx, gz, valid = group_data
            anticommute = (x & gz) ^ (z & gx)
            invalid_support = support & ~valid

            if invalid_support == 0 and (anticommute & valid) == 0:
                terms_list.append((pstring, (x, z), support))
                group_data[1] = gx | x
                group_data[2] = gz | z
                group_data[3] = valid & ~anticommute
                found = True
                break

            all_commute = True
            for _, (x2, z2), _ in terms_list:
                anticommute_full = (x & z2) ^ (z & x2)
                if anticommute_full == 0:
                    continue

                if is_single_block:
                    if _popcount_parity(anticommute_full & single_block_mask):
                        all_commute = False
                        break
                else:
                    for mask in block_masks:
                        if _popcount_parity(anticommute_full & mask):
                            all_commute = False
                            break
                    if not all_commute:
                        break

            if all_commute:
                terms_list.append((pstring, (x, z), support))
                group_data[1] = gx | x
                group_data[2] = gz | z
                group_data[3] = valid & ~anticommute
                found = True
                break

        if not found:
            commuting_sets.append([[(pstring, (x, z), support)], x, z, all_bits_valid])

    if verbosity > 0:
        print(f"Bootstrap: processed first {bootstrap_size} terms, {len(commuting_sets)} groups")

    broadcast_queues = [mp.Queue() for _ in range(num_workers)]
    exclusion_queues = [mp.Queue() for _ in range(num_workers)]
    work_request_queue = mp.Queue()
    work_response_queues = [mp.Queue() for _ in range(num_workers)]
    stop_event = mp.Event()

    chunk_size = 2500
    next_range_start = max(bootstrap_size, 1001)

    initial_ranges = []
    for worker_id in range(num_workers):
        worker_start = next_range_start + worker_id * chunk_size
        worker_end = min(worker_start + chunk_size, N)
        if worker_start >= N:
            break
        initial_ranges.append((worker_start, worker_end))

    next_range_start = initial_ranges[-1][1] if initial_ranges else next_range_start

    if verbosity > 0:
        print(f"Initial ranges: {initial_ranges}")
        print(f"Next range start: {next_range_start}")

    workers = []
    for worker_id, (worker_start, worker_end) in enumerate(initial_ranges):
        if verbosity > 0:
            print(f"Launching worker {worker_id}: initial range [{worker_start}, {worker_end})")

        worker = mp.Process(
            target=_worker,
            args=(worker_id, worker_start, worker_end, cache_path, blocks,
                  broadcast_queues[worker_id], exclusion_queues[worker_id], stop_event,
                  work_request_queue, work_response_queues[worker_id])
        )
        worker.start()
        workers.append(worker)

    exclusions = {}
    last_handshake = bootstrap_size
    total_exclusions_used = 0

    last_group_sizes = {i: len(terms_list) for i, (terms_list, _, _, _) in enumerate(commuting_sets)}
    last_num_groups = len(commuting_sets)
    recently_modified_groups = set()

    for i in range(bootstrap_size, N):
        if verbosity > 0 and i % 100 == 0:
            print(f"Status: On term {i} / {N}, {len(commuting_sets)} groups", end="\r")

        if i - last_handshake >= handshake_interval:
            incremental_updates = []
            for group_id in recently_modified_groups:
                terms_list, gx, gz, vb = commuting_sets[group_id]
                if group_id < last_num_groups:
                    old_size = last_group_sizes[group_id]
                    new_terms = [(x, z) for _, (x, z), _ in terms_list[old_size:]]
                else:
                    new_terms = [(x, z) for _, (x, z), _ in terms_list]
                incremental_updates.append((group_id, gx, gz, vb, new_terms))

            for bq in broadcast_queues:
                bq.put(('update', i, incremental_updates))

            for group_id in recently_modified_groups:
                if group_id < len(commuting_sets):
                    last_group_sizes[group_id] = len(commuting_sets[group_id][0])
            for group_id in range(last_num_groups, len(commuting_sets)):
                last_group_sizes[group_id] = len(commuting_sets[group_id][0])
            last_num_groups = len(commuting_sets)
            recently_modified_groups.clear()

            for eq in exclusion_queues:
                try:
                    new_exclusions = eq.get_nowait()
                    for term_idx, excl_set in new_exclusions.items():
                        exclusions.setdefault(term_idx, set()).update(excl_set)
                    if verbosity > 0:
                        print(f"\nReceived {len(new_exclusions)} exclusions from worker")
                except:
                    pass

            try:
                requesting_worker_id = work_request_queue.get_nowait()
                if next_range_start < N:
                    if next_range_start < 30000:
                        chunk_multiplier = 25
                    elif next_range_start < 40000:
                        chunk_multiplier = 10
                    else:
                        chunk_multiplier = 5

                    current_chunk_size = max(2 * handshake_interval, (chunk_multiplier * handshake_interval) // chunk_divisor)

                    new_start = next_range_start
                    new_end = min(next_range_start + current_chunk_size, N)
                    work_response_queues[requesting_worker_id].put((new_start, new_end))
                    next_range_start = new_end
                    if verbosity > 0:
                        print(f"\nAssigned worker {requesting_worker_id} range [{new_start}, {new_end}) [chunk size: {current_chunk_size}]")
                else:
                    work_response_queues[requesting_worker_id].put((None, None))
                    if verbosity > 0:
                        print(f"\nWorker {requesting_worker_id} completed - no more work")
            except:
                pass

            last_handshake = i

        pstring = terms_ord[i]
        x = int(x_bits_array[i])
        z = int(z_bits_array[i])
        support = int(support_array[i])

        excluded_groups_set = exclusions.get(i, set())
        found = False

        for group_id, group_data in enumerate(commuting_sets):
            if group_id in excluded_groups_set:
                total_exclusions_used += 1
                continue

            terms_list, gx, gz, valid = group_data
            anticommute = (x & gz) ^ (z & gx)
            invalid_support = support & ~valid

            if invalid_support == 0 and (anticommute & valid) == 0:
                terms_list.append((pstring, (x, z), support))
                group_data[1] = gx | x
                group_data[2] = gz | z
                group_data[3] = valid & ~anticommute
                recently_modified_groups.add(group_id)
                found = True
                break

            all_commute = True
            for _, (x2, z2), _ in terms_list:
                anticommute_full = (x & z2) ^ (z & x2)
                if anticommute_full == 0:
                    continue

                if is_single_block:
                    if _popcount_parity(anticommute_full & single_block_mask):
                        all_commute = False
                        break
                else:
                    for mask in block_masks:
                        if _popcount_parity(anticommute_full & mask):
                            all_commute = False
                            break
                    if not all_commute:
                        break

            if all_commute:
                terms_list.append((pstring, (x, z), support))
                group_data[1] = gx | x
                group_data[2] = gz | z
                group_data[3] = valid & ~anticommute
                recently_modified_groups.add(group_id)
                found = True
                break

        if not found:
            new_group_id = len(commuting_sets)
            commuting_sets.append([[(pstring, (x, z), support)], x, z, all_bits_valid])
            recently_modified_groups.add(new_group_id)

    stop_event.set()

    for bq in broadcast_queues:
        try:
            bq.put(('stop', 0, []), timeout=0.1)
        except:
            pass

    for wq in work_response_queues:
        try:
            wq.put((None, None), timeout=0.1)
        except:
            pass

    time.sleep(0.2)

    for worker in workers:
        worker.join(timeout=1.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=0.5)

    for bq in broadcast_queues:
        try:
            bq.close()
            bq.cancel_join_thread()
        except:
            pass
    for eq in exclusion_queues:
        try:
            eq.close()
            eq.cancel_join_thread()
        except:
            pass
    for wq in work_response_queues:
        try:
            wq.close()
            wq.cancel_join_thread()
        except:
            pass
    try:
        work_request_queue.close()
        work_request_queue.cancel_join_thread()
    except:
        pass

    if verbosity > 0:
        print(f"\nCompleted: {len(commuting_sets)} groups")
        print(f"Total exclusions used: {total_exclusions_used}")

    return [[pstring for pstring, _, _ in group_data[0]] for group_data in commuting_sets]
