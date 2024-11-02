from env import AUVEnv 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
from remus100 import plotVehicleStates, plotControls, plot3D
from config import config

env = make_vec_env(AUVEnv)
model = PPO.load("/home/kws/L4DSC_Project_Fall24/PPO_AUV", env=env)
num_steps = 1000

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

figSize1 = [25, 13]  # figure1 size in cm
dpiValue = 150  # figure dpi value

def cm2inch(value):  # inch to cm
    return value / 2.54

simData = np.genfromtxt('PPO_AUV_eval.csv', delimiter=',')

plotVehicleStates(simTime, simData, 2)                    
plotControls(simTime, simData, 3)
# plot3D(simData, target_positions, 50, 10, '3D_animation.gif', 3)  

plt.show()