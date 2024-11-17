from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import traceback
import math

from callbacks import WandBVideoCallback, checkpoint_callback 
from config import config 
from env import AUVEnv 

run = wandb.init(
    project="AUV_env_SAC",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# Save config, env, and rewards file for each training run to wandb
wandb.save( "./config.py")
wandb.save("./rewards.py")
wandb.save("./env.py")

env = AUVEnv()
env = Monitor(env)

model = SAC(
    config["policy_cls"],
    env,
    verbose=config["verbose"],
    device=config["device"],
    tensorboard_log=f"runs/{run.id}",
)
try:
    model.learn(
        total_timesteps=config["max_steps"],
        callback=[
            WandbCallback(
                verbose=2,
                model_save_path=None,  # f"models/{run.id}"
                model_save_freq=0,  # 100
                gradient_save_freq=0,  # 100
            ),
            # WandBVideoCallback(),

            checkpoint_callback
        ],
    )
    save_dir = "./final_model/"
    model.save(save_dir + "SAC_AUV")

except:
    traceback.print_exc()
run.finish()
