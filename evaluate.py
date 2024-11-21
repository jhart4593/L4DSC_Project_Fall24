from env import AUVEnv 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
from remus100 import plotVehicleStates, plotControls, plot3D, plot_controls_2D, plot_attitude_2D
from config import config
from eval_config import eval_config

env = make_vec_env(AUVEnv)
model = PPO.load("./rl_model_25000000_steps.zip", env=env)
num_steps = 5000

filename = "PPO_AUV_eval.csv"
f = open(filename, "w+")
f.close()

episode_rewards = [0.0]
obs = env.reset()
simTime = []

for i in range(num_steps):
    # _states are only useful when using LSTM policies
    t = i * config["sim_dt"]
    simTime.append(t)

    action, _  = model.predict(obs)
    obs, rew, done, info = env.step(action)
    
    # Stats
    episode_rewards[-1] += rew
    if done:
        break

simData = np.genfromtxt('PPO_AUV_eval.csv', delimiter=',')

simTime = []
for i in range(simData.shape[0]):
    t = i * config["sim_dt"]
    simTime.append(t)

target_positions = eval_config["path"]

plotVehicleStates(simTime, simData, 'PPO_AUV_eval_states.png', 2)                    
plotControls(simTime, simData, 'PPO_AUV_eval_controls.png', 3)
plot3D(simData, target_positions, 50, 10, 'PPO_AUV_eval_3D.gif', 4)  
plot_controls_2D(simData, 50, 'PPO_AUV_eval_controls_2D.gif', 5)
plot_attitude_2D(simData, 50, 'PPO_AUV_eval_attitude_2D.gif', 6)

# plt.show()