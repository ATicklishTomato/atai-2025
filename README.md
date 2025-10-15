# Fork of atai-2025
Advanced Topics in AI course @TU/e repository

## Group
- Dirk Burgers
- Seppe Hannen
- Luuk Wubben


## Development
Install requirements. Make sure to use the PyTorch wheel for your CUDA version. Check your CUDA by running `nvcc --version` in the terminal.
```bash
pip install -r requirements.txt
```

## Running assignment 2 code
```
usage: run.py [-h] [--problem {cfd,boids}] [--sweep] [--sweep_runs SWEEP_RUNS] [--run_tags RUN_TAGS [RUN_TAGS ...]] [--epochs EPOCHS] [--patience PATIENCE] [--batch_size BATCH_SIZE] [--lr LR] [--predict_frames PREDICT_FRAMES]
              [--history_frames HISTORY_FRAMES] [--condition_on {prior,vector_field,both}] [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--ch_mults CH_MULTS [CH_MULTS ...]] [--sigma SIGMA] [--euler_steps EULER_STEPS]
              [--device DEVICE] [--use_tqdm] [--verbose {10,20,30,40}] [--save] [--load] [--skip_train] [--skip_test] [--no_save_figures] [--wandb_api_key WANDB_API_KEY]

Train and test a neural network on either a cfd or a boids dataset

options:
  -h, --help            show this help message and exit
  --problem {cfd,boids}
                        Type of problem to train and test the respective model on. Default is cfd.
  --sweep               Run a hyperparameter sweep. Default is False. Note: This will override any arguments passed related to sweep parameters
  --sweep_runs SWEEP_RUNS
                        Number of random runs to perform in the hyperparameter sweep. Default is 10
  --run_tags RUN_TAGS [RUN_TAGS ...]
                        Tags to add to the Weights and Biases run. Default is empty list. Example usage: --run_tags tag1 tag2
  --epochs EPOCHS       Number of epochs to train for. Default is 200
  --patience PATIENCE   Number of epochs with no improvement on validation loss before stopping training early. Default is 20
  --batch_size BATCH_SIZE
                        Batch size for training. Default is 4
  --lr LR               Learning rate for training. Default is 1e-4
  --predict_frames PREDICT_FRAMES
                        Number of frames to predict. Default is 20
  --history_frames HISTORY_FRAMES
                        Number of history frames to condition on. Default is 4
  --condition_on {prior,vector_field,both}
                        CFD ONLY - Type of conditioning to apply using history frames. Default is "prior"
  --hidden_size HIDDEN_SIZE
                        Base hidden size for the model. Can be multiplied for deeper layers. Default is 64
  --num_layers NUM_LAYERS
                        Number of layers in the model. Default is 3
  --ch_mults CH_MULTS [CH_MULTS ...]
                        Channel multipliers for each layer in the model. Default is [1, 2, 2]. Example usage: --ch_mults 1 2 2
  --sigma SIGMA         Noise level for flow matching model and variance on conditioned prior. Default is 0.015
  --euler_steps EULER_STEPS
                        Number of Euler steps to use during inference. Default is 20
  --device DEVICE       PyTorch device to train on. Default is cuda
  --use_tqdm            Use tqdm progress bars during training. Default is False
  --verbose {10,20,30,40}
                        Verbosity level for logging. Options are for DEBUG, INFO, WARNING, and ERROR, respectively. Default is INFO
  --save                Save the model and optimizer state_dicts (if applicable) after training. Default is False
  --load                Load the stored model and optimizer state_dicts (if applicable) before training and skip training. Default is False
  --skip_train          Skip training and only evaluate the model. Default is False
  --skip_test           Skip testing and only train the model. Default is False
  --no_save_figures     Save any figures that are generated during testing. Default is True
  --wandb_api_key WANDB_API_KEY
                        Your personal API key for Weights and Biases. Default is None. Alternatively, you can leave this empty and store the key in a file in the root of this script called "wandb.login". This file will be
                        ignored by git. NOTE: Make sure to keep this key private and secure. Do not share it or upload it to a public repository.
```

## Snellius supercomputer usage
To run the code on the Snellius supercomputer, you need to copy the assignment2 directory to the supercomputer using `scp`.
```bash
scp -r assignment2 <username>@snellius.surf.nl:~/.
```

**IMPORTANT:** Do not forget to include a `wandb.login` file in the project root with your Weights and Biases API key. Using the CLI flag on Snellius isn't advised.

After copying the code, you can run the code using the following command:
```bash
dos2unix snellius_job.bash # to convert DOS line breaks to UNIX line breaks
sbatch snellius_job.bash
```
**NOTE:** Before running, ensure you've updated the relevant `SBATCH` flags in the `snellius_job.bash` script, as well as the python execution command. (This can be ignored for vanilla runs, as it currently works, we can modify this later)

You can then check the status of jobs started by your user using the `squeue` command.
```bash
squeue -u <username>
```

You can then check the status of the specific job with the `-j` flag.
```bash
squeue -j <job_id>
```

You can cancel the job using the `scancel` command.
```bash
scancel <job_id>
```

### script setup
The `snellius_job.bash` script is set up to run the code on the Snellius supercomputer.
To pass parameters, we use `#SBATCH` flags in the script:
- `#SBATCH --time=2:00:00` to specify the maximum time the job can run
- `#SBATCH -p gpu_mig` to specify the partition to use. `gpu_mig` uses GPU partitions. `gpu`uses whole GPUs.
- `#SBATCH -N 1` to specify the number of nodes to use
- `#SBATCH --tasks-per-node 1` to specify the number of tasks per node
- `#SBATCH --gpus=1` to specify the number of GPUs to use
- `#SBATCH --output=R-%x.%j.out` to specify the output file

More information on how to set up the script for different environments, e.g. using one or multiple CPUs, can be found in the [Snellius documentation](https://servicedesk.surf.nl/wiki/display/WIKI/Example+job+scripts).
