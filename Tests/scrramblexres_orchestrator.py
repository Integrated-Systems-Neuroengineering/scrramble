"""
Orchastrator module for scrramblexres tests on CIFAR10.
"""
import subprocess
import itertools
import os
import sys
import time
from datetime import date
from pathlib import Path
import pandas as pd

today = date.today().isoformat()

## Define Sweep parameters
CONNECTION_DENSITIES = [0.01, 0.05, 0.1, 0.15, 0.5, 0.9, 1.0]
SLOT_SIZES = [64, 16, 8]
NUM_RESAMPLES = 3

## Fixed Parameters
CAPSULE_SIZE = 256
CAPSULE_LAYERS = [50, 10]
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TRAIN_STEPS = int(5e4)
EVAL_EVERY = int(5e3)

## Paths
SCRIPT_PATH = Path(__file__).parent / "sweep_test_scrramblexres.py"
RESULTS_CSV = f"/Volumes/export/isn/vikrant/Data/scrramble/logs/scrramble_resnet20_cifar10_sweep_results_{today}.csv"
LOG_DIR = Path(f"/local_disk/vikrant/scrramble/scrramblexres_sweep_logs/logs_scrramble_resnet20_cifar10_sweep_{today}")

LOG_DIR.mkdir(parents=True, exist_ok=True)  # create log directory if not exists

## Check if the config was already run
def check_if_already_run(conn_density, slot_size, repeat):
    """Check if this configuration was already completed"""
    if not os.path.exists(RESULTS_CSV):
        return False
    
    try:
        df = pd.read_csv(RESULTS_CSV)
        mask = (
            (df['connection_density'] == conn_density) &
            (df['slot_size'] == slot_size) &
            (df['repeat'] == repeat)
        )
        return mask.any()
    except:
        return False
    


## run a single config
def run_single_config(connection_density, slot_size, resample_idx, config_idx, total_configs):
    """
    Runs a single config of ScRRAMBLE-RES on CIFAR10.
    """

    # check if already completed
    if check_if_already_run(connection_density, slot_size, resample_idx):
        print(f"Config {config_idx+1}/{total_configs} already completed. Skipping.")
        return True
    
    # Build the command
    cmd = [
        sys.executable, 
        str(SCRIPT_PATH),
        "--connection_density", str(connection_density),
        "--slot_size", str(slot_size),
        "--resample", str(resample_idx),
        "--train_steps", str(TRAIN_STEPS),
        "--eval_every", str(EVAL_EVERY),
    ]

    # set up the log file
    log_file = LOG_DIR / f"conn{connection_density}_slot{slot_size}_rep{resample_idx}.log"

    # Run subprocess
    start_time = time.time()

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60*60*3 # 3 hours timeout
            )

        elapsed = time.time() - start_time

        if result.returncode == 0: # successful execution
            print(f"SUCCESS. Completed config {config_idx+1}/{total_configs} in {elapsed/60:.2f} minutes.")
            return True
        else:
            print(f"FAILED! Config {config_idx+1}/{total_configs} exited with code {result.returncode}.")
            print(f"\t SEE LOG: {log_file}")
            return False
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 3 hours")
        return False
    
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    

# main function to run all configs
def main():
    configs = list(itertools.product(CONNECTION_DENSITIES, SLOT_SIZES, range(NUM_RESAMPLES)))
    total_configs = len(configs)

    print(f"\n{'='*80}")
    print(f"PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Total configurations: {total_configs}")
    print(f"Connection densities: {CONNECTION_DENSITIES}")
    print(f"Slot sizes: {SLOT_SIZES}")
    print(f"Resamples per config: {NUM_RESAMPLES}")
    print(f"Results will be saved to: {RESULTS_CSV}")
    print(f"Logs will be saved to: {LOG_DIR}")
    print(f"{'='*80}\n")

    # Running sweep
    successful = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for i, (p, ls, r) in enumerate(configs, 1):

        successful_run = run_single_config(connection_density=p, slot_size=ls, resample_idx=r, config_idx=i-1, total_configs=total_configs)

        if successful_run:
            successful += 1
        else:
            failed += 1

        # Print progress summary
        elapsed = time.time() - start_time
        avg_time = elapsed / (i - skipped) if (i - skipped) > 0 else 0
        remaining = avg_time * (total_configs - i)
        
        print(f"\nProgress: {successful} successful, {failed} failed, {skipped} skipped")
        print(f"Estimated time remaining: {remaining/3600:.1f} hours\n")

        # Final summary
    total_time = time.time() - start_time
    print(f"\n{'++'*80}")
    print(f"SWEEP COMPLETE!")
    print(f"{'++'*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful: {successful}/{total_configs}")
    print(f"Failed: {failed}/{total_configs}")
    print(f"Skipped: {skipped}/{total_configs}")

    if successful > 0:
        print(f"Results saved to: {RESULTS_CSV}")
    else:
        print(f"No results saved (no successful runs)")
    
    print(f"{'='*80}\n")

    
    # Load and display summary
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        print("\nResults summary:")
        summary = df.groupby(['connection_density', 'slot_size'])['test_accuracy'].agg(['mean', 'std', 'count'])
        print(summary)


if __name__ == "__main__":
    main()





