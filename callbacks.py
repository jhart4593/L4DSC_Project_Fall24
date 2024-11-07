import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from config import config


class WandBVideoCallback(BaseCallback):
    def __init__(self, verbose=0) -> None:
        super(WandBVideoCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.training_env.get_attr('plot') is not None:
            wandb.log({"plot": wandb.Image(self.training_env.get_attr('plot'))})
        return True

    # def _on_step(self) -> bool:

    #     return True

    # def _on_rollout_end(self) -> None:
        
    #     wandb.log({"plot": wandb.Image(self.training_env.env_method('render'))})

    #     pass

# Save checkpoint every 1000 steps
save_freq = config["model_save_freq"]
checkpoint_callback = CheckpointCallback(
    save_freq = max(save_freq // config["num_envs"], 1),
    save_path="./model_logs/",
    name_prefix="rl_model"
)