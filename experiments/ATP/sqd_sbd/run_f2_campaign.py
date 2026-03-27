#!/usr/bin/env python3
"""Mini job scheduler for the f2 campaign on a multi-GPU workstation.

Runs up to N_GPUS jobs in parallel, one per GPU. When a job finishes,
the next queued job starts on the freed GPU. All output goes to log files.

Usage:
    nohup python run_f2_campaign.py &> campaign_manager.log &
    disown

Then disconnect SSH. Check progress with:
    tail -f campaign_manager.log
    ls -lt logs/f2_*.out | head
"""
import subprocess
import os
import sys
import time
import signal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_GPUS = 6
DISPATCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "dispatch_campaign_f2.py")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
PYTHON = sys.executable

# Job IDs to run (0-52 = all 53 f2 jobs)
JOB_IDS = list(range(53))

# Environment setup
ENV = os.environ.copy()
ENV["MPIRUN"] = os.path.expanduser(
    "~/nvhpc/Linux_x86_64/25.1/comm_libs/mpi/bin/mpirun")
ENV["OMP_NUM_THREADS"] = "1"
# Disable core dumps
ENV["NVIDIA_COREDUMP_PIPE"] = ""

# Add NVHPC libs to PATH and LD_LIBRARY_PATH
NVHPC = os.path.expanduser("~/nvhpc/Linux_x86_64/25.1")
ENV["PATH"] = f"{NVHPC}/compilers/bin:{NVHPC}/comm_libs/mpi/bin:{ENV.get('PATH', '')}"
ENV["LD_LIBRARY_PATH"] = (
    f"{NVHPC}/compilers/lib:{NVHPC}/comm_libs/mpi/lib:{NVHPC}/cuda/lib64:"
    f"{ENV.get('LD_LIBRARY_PATH', '')}"
)

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Disable core dumps for this process and children
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    queue = list(JOB_IDS)
    # gpu_id -> (process, job_id, start_time, log_out, log_err)
    running = {}
    completed = []
    failed = []

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    log(f"f2 campaign: {len(queue)} jobs, {N_GPUS} GPUs")
    log(f"Dispatch script: {DISPATCH_SCRIPT}")
    log(f"Log directory: {LOG_DIR}")

    def launch(gpu_id, job_id):
        out_path = os.path.join(LOG_DIR, f"f2_job{job_id:02d}_gpu{gpu_id}.out")
        err_path = os.path.join(LOG_DIR, f"f2_job{job_id:02d}_gpu{gpu_id}.err")
        fout = open(out_path, "w")
        ferr = open(err_path, "w")

        job_env = ENV.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [PYTHON, DISPATCH_SCRIPT, str(job_id)]
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, env=job_env)

        running[gpu_id] = (proc, job_id, time.time(), fout, ferr)
        log(f"  GPU {gpu_id}: started job {job_id} (pid {proc.pid})")

    def check_done():
        done_gpus = []
        for gpu_id, (proc, job_id, t0, fout, ferr) in running.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                fout.close()
                ferr.close()
                if ret == 0:
                    log(f"  GPU {gpu_id}: job {job_id} DONE ({elapsed/3600:.1f}h)")
                    completed.append(job_id)
                else:
                    log(f"  GPU {gpu_id}: job {job_id} FAILED (rc={ret}, {elapsed/3600:.1f}h)")
                    failed.append(job_id)
                done_gpus.append(gpu_id)
        for g in done_gpus:
            del running[g]

    # Graceful shutdown on SIGTERM/SIGINT
    shutdown = False
    def handle_signal(signum, frame):
        nonlocal shutdown
        log(f"Received signal {signum}, finishing current jobs then stopping...")
        shutdown = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main loop
    while queue or running:
        check_done()

        # Launch new jobs on free GPUs
        while queue and not shutdown:
            free_gpus = [g for g in range(N_GPUS) if g not in running]
            if not free_gpus:
                break
            gpu_id = free_gpus[0]
            job_id = queue.pop(0)
            launch(gpu_id, job_id)

        if not running:
            break

        time.sleep(30)

    # Summary
    log(f"\n{'='*60}")
    log(f"Campaign complete: {len(completed)} done, {len(failed)} failed, {len(queue)} skipped")
    if failed:
        log(f"Failed jobs: {failed}")
    if queue:
        log(f"Remaining (not started): {queue}")

if __name__ == "__main__":
    main()
