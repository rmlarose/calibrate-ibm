#!/usr/bin/env python3
"""Dispatch SQD campaign jobs by SLURM array task ID.

Usage:
    python dispatch_campaign.py $SLURM_ARRAY_TASK_ID
    python dispatch_campaign.py --dry-run   # print all 65 commands
"""
import os
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE = "/mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments"
SQD_SCRIPT = os.path.join(BASE, "ATP/sqd_sbd/run_sqd_sbd.py")
SBD_EXE = "/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"

FRAGMENT_CONFIG = {
    'metaphosphate-2026': {
        'circuit_dir':    os.path.join(BASE, "metaphosphate/circuits"),
        'hamiltonian_dir': os.path.join(BASE, "metaphosphate/hamiltonians"),
        'results_dir':    os.path.join(BASE, "metaphosphate/results"),
        'output_base':    os.path.join(BASE, "metaphosphate/sqd_sbd/results"),
        'data_dir':       os.path.join(BASE, "metaphosphate/sqd_sbd/data"),
    },
    'atp_0_be2_f4': {
        'circuit_dir':    os.path.join(BASE, "ATP/circuits"),
        'hamiltonian_dir': os.path.join(BASE, "ATP/hamiltonians"),
        'results_dir':    os.path.join(BASE, "ATP/results"),
        'output_base':    os.path.join(BASE, "ATP/sqd_sbd/f4/results"),
        'data_dir':       os.path.join(BASE, "ATP/sqd_sbd/f4/data"),
    },
    'atp_0_be2_f18': {
        'circuit_dir':    os.path.join(BASE, "ATP/circuits"),
        'hamiltonian_dir': os.path.join(BASE, "ATP/hamiltonians"),
        'results_dir':    os.path.join(BASE, "ATP/results"),
        'output_base':    os.path.join(BASE, "ATP/sqd_sbd/f18/results"),
        'data_dir':       os.path.join(BASE, "ATP/sqd_sbd/f18/data"),
    },
}

# SC counts key -> pickle filename
SC_COUNTS_FILES = {
    'random_hamming': 'counts_random_hamming_50000.pkl',
    'random_iid':     'counts_random_iid_50000.pkl',
    'sc_10k':         'counts_10000.pkl',
    'sc_50k':         'counts_50000.pkl',
    'sc_100k':        'counts_100000.pkl',
}

MAX_ITERATIONS = 5000

# ---------------------------------------------------------------------------
# Job table builder
# ---------------------------------------------------------------------------

def build_job_table():
    """Build the complete job table (65 jobs for MIG campaign)."""
    jobs = []

    def add(fragment, output_subdir, resume, sc_counts, adapts):
        jobs.append({
            'fragment': fragment,
            'output_subdir': output_subdir,
            'resume': resume,
            'sc_counts': sc_counts,
            'adapts': list(adapts),
        })

    def sc_name(adapts):
        """Build semiclassical directory name from ADAPT iteration list."""
        if not adapts:
            return 'semiclassical_0'
        return 'semiclassical_0_' + '_'.join(map(str, adapts))

    # =======================================================================
    # Metaphosphate (22 orb, 32 elec) — 20 jobs
    # =======================================================================
    META = 'metaphosphate-2026'
    META_ADAPTS = [1, 2, 3, 4, 5, 10]

    # Hardware: 6 unconverged, resume (jobs 0-5)
    for a in [4, 5, 10]:
        add(META, f'hardware/singleton_{a}', True, None, [a])
    for cum in [[1,2,3,4], [1,2,3,4,5], [1,2,3,4,5,10]]:
        add(META, f'hardware/cumulative_{"_".join(map(str, cum))}', True, None, cum)

    # Random Hamming 50k: 7 jobs (jobs 6-12)
    for i in range(len(META_ADAPTS) + 1):
        adapts = META_ADAPTS[:i]
        add(META, f'semiclassical/results_random_hamming_50k/{sc_name(adapts)}',
            True, 'random_hamming', adapts)

    # Random IID 50k: 7 jobs (jobs 13-19)
    for i in range(len(META_ADAPTS) + 1):
        adapts = META_ADAPTS[:i]
        add(META, f'semiclassical/results_random_iid_50k/{sc_name(adapts)}',
            True, 'random_iid', adapts)

    # =======================================================================
    # f4 (32 orb, 32 elec) — 24 jobs
    # =======================================================================
    F4 = 'atp_0_be2_f4'
    F4_ADAPTS = [1, 2, 3, 4, 5, 10, 20, 25, 30, 40]

    # Hardware: 3 unconverged, resume (jobs 20-22)
    add(F4, 'hardware/singleton_30', True, None, [30])
    add(F4, 'hardware/cumulative_1_2_3_4_5_10_20_25_30', True, None,
        [1,2,3,4,5,10,20,25,30])
    add(F4, 'hardware/cumulative_1_2_3_4_5_10_20_25_30_40', True, None,
        [1,2,3,4,5,10,20,25,30,40])

    # SC 10k: 2 unconverged, resume (jobs 23-24)
    for end in [9, 10]:  # F4_ADAPTS[:9]=[1..30], [:10]=all
        adapts = F4_ADAPTS[:end]
        add(F4, f'semiclassical/results_10k/{sc_name(adapts)}',
            True, 'sc_10k', adapts)

    # SC 50k: 4 unconverged, resume (jobs 25-28)
    for end in [7, 8, 9, 10]:  # [:7]=[1..20], [:8]=[1..25], [:9]=[1..30], [:10]=all
        adapts = F4_ADAPTS[:end]
        add(F4, f'semiclassical/results_50k/{sc_name(adapts)}',
            True, 'sc_50k', adapts)

    # SC 100k: 3 unconverged, resume (jobs 29-31)
    for end in [8, 9, 10]:  # [:8]=[1..25], [:9]=[1..30], [:10]=all
        adapts = F4_ADAPTS[:end]
        add(F4, f'semiclassical/results_100k/{sc_name(adapts)}',
            True, 'sc_100k', adapts)

    # Random Hamming 50k: 4 unconverged, resume (jobs 32-35)
    for end in [7, 8, 9, 10]:
        adapts = F4_ADAPTS[:end]
        add(F4, f'semiclassical/results_random_hamming_50k/{sc_name(adapts)}',
            True, 'random_hamming', adapts)

    # Random IID 50k: 8 unconverged, resume (jobs 36-43)
    for end in [1, 2, 3, 4, 7, 8, 9, 10]:
        adapts = F4_ADAPTS[:end]
        add(F4, f'semiclassical/results_random_iid_50k/{sc_name(adapts)}',
            True, 'random_iid', adapts)

    # =======================================================================
    # f18 (57 orb, 58 elec) — 21 jobs
    # =======================================================================
    F18 = 'atp_0_be2_f18'
    F18_ADAPTS = [1, 2, 3, 4, 5]

    # Hardware: 9 jobs (jobs 44-52)
    for a in F18_ADAPTS:
        add(F18, f'hardware/singleton_{a}', True, None, [a])
    for i in range(2, len(F18_ADAPTS) + 1):
        cum = F18_ADAPTS[:i]
        add(F18, f'hardware/cumulative_{"_".join(map(str, cum))}', True, None, cum)

    # Random Hamming 50k: 6 jobs (jobs 53-58)
    for i in range(len(F18_ADAPTS) + 1):
        adapts = F18_ADAPTS[:i]
        add(F18, f'semiclassical/results_random_hamming_50k/{sc_name(adapts)}',
            True, 'random_hamming', adapts)

    # Random IID 50k: 6 jobs (jobs 59-64)
    for i in range(len(F18_ADAPTS) + 1):
        adapts = F18_ADAPTS[:i]
        add(F18, f'semiclassical/results_random_iid_50k/{sc_name(adapts)}',
            True, 'random_iid', adapts)

    return jobs


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(job):
    """Build the full command list for a job."""
    cfg = FRAGMENT_CONFIG[job['fragment']]
    output_dir = os.path.join(cfg['output_base'], job['output_subdir'])

    cmd = [
        sys.executable, SQD_SCRIPT,
        '--fragment', job['fragment'],
        '--circuit_dir', cfg['circuit_dir'],
        '--hamiltonian_dir', cfg['hamiltonian_dir'],
        '--results_dir', cfg['results_dir'],
        '--sbd_exe', SBD_EXE,
        '--output_dir', output_dir,
        '--max_iterations', str(MAX_ITERATIONS),
    ]

    if job['resume']:
        cmd.append('--resume')

    if job['sc_counts']:
        counts_file = os.path.join(cfg['data_dir'], SC_COUNTS_FILES[job['sc_counts']])
        cmd.extend(['--semiclassical_counts', counts_file])

    if job['adapts']:
        cmd.extend(str(a) for a in job['adapts'])

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    jobs = build_job_table()
    assert len(jobs) == 65, f"Expected 65 jobs, got {len(jobs)}"

    if '--dry-run' in sys.argv:
        for i, job in enumerate(jobs):
            cmd = build_command(job)
            frag_short = job['fragment'].replace('atp_0_be2_', '').replace('metaphosphate-2026', 'meta')
            resume_tag = ' [resume]' if job['resume'] else ''
            print(f"[{i:2d}] {frag_short:>5s} {job['output_subdir']}{resume_tag}")
            print(f"     {' '.join(cmd)}")
            print()
        print(f"Total: {len(jobs)} jobs")
        return

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} TASK_ID  or  {sys.argv[0]} --dry-run",
              file=sys.stderr)
        sys.exit(1)

    task_id = int(sys.argv[1])
    if task_id < 0 or task_id >= len(jobs):
        print(f"Error: task_id {task_id} out of range [0, {len(jobs)-1}]",
              file=sys.stderr)
        sys.exit(1)

    job = jobs[task_id]
    cmd = build_command(job)
    print(f"Job {task_id}: {job['fragment']} / {job['output_subdir']}")
    print(f"Resume: {job['resume']}, SC counts: {job['sc_counts']}")
    print(f"Command: {' '.join(cmd)}")
    print(flush=True)

    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
