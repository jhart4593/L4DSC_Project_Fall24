from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import traceback
import math

from callbacks import WandBVideoCallback, checkpoint_callback  
from config import config 
from env import AUVEnv 

run = wandb.init(
    project="AUV_env",
    config=config,
    sync_tensorboard=True,
    # monitor_gym=True,
    save_code=True,
)

# Save config, env, and rewards file for each training run to wandb
wandb.save( "./config.py")
wandb.save("./rewards.py")
wandb.save("./env.py")

# env = AUVEnv(render_mode="human")
# def make_env():
#     env = AUVEnv()
#     env = Monitor(env)  # record stats such as returns
#     return env
# env = DummyVecEnv([make_env])
env = make_vec_env(AUVEnv,n_envs=config["num_envs"])

model = PPO(
    config["policy_cls"],
    env,
    policy_kwargs=config["policy_kwargs"],
    verbose=config["verbose"],
    device=config["device"],
    tensorboard_log=f"runs/{run.id}",
    n_steps=config["rollout_steps"],
    batch_size=config["minibatch_size"]
)
try:
    model.learn(
        total_timesteps=config["max_steps"],
        callback=[
            WandbCallback(
                verbose=config["verbose"],
                model_save_path=None,  # f"models/{run.id}"
                model_save_freq=0,  # 100
                gradient_save_freq=0,  # 100
            ),

            # WandBVideoCallback(),

            checkpoint_callback
        ],
    )
    save_dir = "./final_model/"
    model.save(save_dir + "PPO_AUV")

# except Exception as e:
#     print(f"Exception: {e}")
except:
    traceback.print_exc()
run.finish()