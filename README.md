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
