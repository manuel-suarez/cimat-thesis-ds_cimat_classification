import os

# Define array of SLURM variables
slurm_vars = {
    "array_task_id": int(os.getenv("SLURM_ARRAY_TASK_ID", 1)),
    "array_job_id": int(os.getenv("SLURM_ARRAY_JOB_ID", 1)),
}
