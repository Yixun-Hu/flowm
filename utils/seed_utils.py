"""Opt-in reproducibility for model initialization and data workers."""
def seed_run(seed):
    if seed is not None:
        from lightning.pytorch import seed_everything
        seed_everything(int(seed), workers=True)
