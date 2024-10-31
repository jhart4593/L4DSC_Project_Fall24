import wandb
from stable_baselines3.common.callbacks import BaseCallback


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