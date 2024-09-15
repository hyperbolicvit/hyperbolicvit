#!/bin/bash

# Define the launchrun function
function launchrun() {
    gpus="$1"
    logfile="$2"
    shift 2  # Remove the first two arguments (gpus and logfile)

    # Generate a semi-unique session ID using date
    session_id="session_$(date +"%Y%m%d%H%M%S")"

    # Create a tmux session, run the command inside it, and detach
    tmux new-session -d -s "$session_id" "CUDA_VISIBLE_DEVICES=$gpus $* > $logfile 2>&1"

    echo "Session ID: $session_id"
}

# Example usage of the launchrun function
gpus="0,1,2,3,4,5,6,7"  # Specify which GPUs to use (e.g., GPU 0, 1, 2, 3)
logfile="test_log.txt"  # Specify the log file
trainfile="test_ddp.py"  # The training script

# Set the number of processes for distributed training (equal to the number of GPUs)
num_processes=8

# Launch the distributed training using torch.multiprocessing
launchrun "$gpus" "$logfile" torchrun --nproc_per_node=$num_processes $trainfile
