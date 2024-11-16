from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
import traceback

from callbacks import WandBVideoCallback 
from config import config 
from env import AUVEnv 

run = wandb.init(
    project="AUV_env",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

env = AUVEnv(render_mode="human")
# env = make_vec_env(AUVEnv,n_envs=1)

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
                verbose=config["verbose"],
                model_save_path=f"models/{run.id}",  # f"models/{run.id}"
                model_save_freq=100,  # 100
                gradient_save_freq=100,  # 100
            ),
            # WandBVideoCallback(),
        ],
    )
# except Exception as e:
#     print(f"Exception: {e}")
except:
    traceback.print_exc()
run.finish()
