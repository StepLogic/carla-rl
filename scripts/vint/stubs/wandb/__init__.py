"""Stand-in for wandb in the ViNT CARLA fine-tune (config use_wandb: False).

The cluster's wandb 0.14.2 cannot be imported there (its dependency `click` is
missing), and ViNT's training code imports wandb at module level although it
only calls it behind use_wandb / image logging, both off. Importing this works;
using any attribute raises, so an unexpected wandb call fails loudly instead of
silently logging nowhere. slurm/vint_train.sbatch puts it first on PYTHONPATH.
"""


def __getattr__(name):
    raise RuntimeError(f"wandb.{name} was used, but this is the wandb stub "
                       "(scripts/vint/stubs): keep use_wandb False and image logging off")
