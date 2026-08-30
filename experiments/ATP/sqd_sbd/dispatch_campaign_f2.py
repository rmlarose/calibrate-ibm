#!/usr/bin/env python3
"""Dispatch f2 SQD campaign jobs by SLURM array task ID.

f2 (44 orb, 44 elec, 88 qubits) is too large for MIG slices.
Run on a full GPU (remote server or full A100/H200 partition).

Usage:
    python dispatch_campaign_f2.py $SLURM_ARRAY_TASK_ID
    python dispatch_campaign_f2.py --dry-run   # print all 53 commands
"""
import os
import socket
import sys

# ---------------------------------------------------------------------------
# Constants — auto-detect ICER vs remote server
# ---------------------------------------------------------------------------
_hostname = socket.gethostname()
if any(_hostname.startswith(p) for p in ('nal-', 'acm-', 'agg-', 'dev-')):
    # ICER
    BASE = "/mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments"
    SBD_EXE = "/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"
else:
    # Remote (rotskoff)
    BASE = "/home/rowlan91/ben/calibrate-ibm/experiments"
    SBD_EXE = "/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
SQD_SCRIPT = os.path.join(BASE, "ATP/sqd_sbd/run_sqd_sbd.py")

FRAGMENT = 'atp_0_be2_f2'
CIRCUIT_DIR = os.path.join(BASE, "ATP/circuits")
HAMILTONIAN_DIR = os.path.join(BASE, "ATP/hamiltonians")
RESULTS_DIR = os.path.join(BASE, "ATP/results")
OUTPUT_BASE = os.path.join(BASE, "ATP/sqd_sbd/f2/results")
DATA_DIR = os.path.join(BASE, "ATP/sqd_sbd/f2/data")

SC_COUNTS_FILES = {
    'random_hamming': 'counts_random_hamming_50000.pkl',
    'random_iid':     'counts_random_iid_50000.pkl',
}

MAX_ITERATIONS = 5000
F2_ADAPTS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50]

# ---------------------------------------------------------------------------
# Job table builder
# ---------------------------------------------------------------------------

def build_job_table():
    """Build the complete f2 job table (53 jobs)."""
    jobs = []

    def add(output_subdir, resume, sc_counts, adapts):
        jobs.append({
            'output_subdir': output_subdir,
            'resume': resume,
            'sc_counts': sc_counts,
            'adapts': list(adapts),
        })

    def sc_name(adapts):
        if not adapts:
            return 'semiclassical_0'
        return 'semiclassical_0_' + '_'.join(map(str, adapts))

    # =======================================================================
    # Hardware: 25 jobs (resume=True for all; no-ops if no checkpoint)
    # =======================================================================

    # 13 singletons (jobs 0-12)
    for a in F2_ADAPTS:
        add(f'hardware/singleton_{a}', True, None, [a])

    # 12 cumulatives (jobs 13-24)
    for i in range(2, len(F2_ADAPTS) + 1):
        cum = F2_ADAPTS[:i]
        add(f'hardware/cumulative_{"_".join(map(str, cum))}', True, None, cum)

    # =======================================================================
    # Random Hamming 50k: 14 jobs (jobs 25-38)
    # =======================================================================
    for i in range(len(F2_ADAPTS) + 1):
        adapts = F2_ADAPTS[:i]
        add(f'semiclassical/results_random_hamming_50k/{sc_name(adapts)}',
            True, 'random_hamming', adapts)

    # =======================================================================
    # Random IID 50k: 14 jobs (jobs 39-52)
    # =======================================================================
    for i in range(len(F2_ADAPTS) + 1):
        adapts = F2_ADAPTS[:i]
        add(f'semiclassical/results_random_iid_50k/{sc_name(adapts)}',
            True, 'random_iid', adapts)

    return jobs


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(job):
    output_dir = os.path.join(OUTPUT_BASE, job['output_subdir'])

    cmd = [
        sys.executable, SQD_SCRIPT,
        '--fragment', FRAGMENT,
        '--circuit_dir', CIRCUIT_DIR,
        '--hamiltonian_dir', HAMILTONIAN_DIR,
        '--results_dir', RESULTS_DIR,
        '--sbd_exe', SBD_EXE,
        '--output_dir', output_dir,
        '--max_iterations', str(MAX_ITERATIONS),
    ]

    if job['resume']:
        cmd.append('--resume')

    if job['sc_counts']:
        counts_file = os.path.join(DATA_DIR, SC_COUNTS_FILES[job['sc_counts']])
        cmd.extend(['--semiclassical_counts', counts_file])

    if job['adapts']:
        cmd.extend(str(a) for a in job['adapts'])

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    jobs = build_job_table()
    assert len(jobs) == 53, f"Expected 53 jobs, got {len(jobs)}"

    if '--dry-run' in sys.argv:
        for i, job in enumerate(jobs):
            cmd = build_command(job)
            resume_tag = ' [resume]' if job['resume'] else ''
            print(f"[{i:2d}] f2 {job['output_subdir']}{resume_tag}")
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
    print(f"Job {task_id}: f2 / {job['output_subdir']}")
    print(f"Resume: {job['resume']}, SC counts: {job['sc_counts']}")
    print(f"Command: {' '.join(cmd)}")
    print(flush=True)

    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
