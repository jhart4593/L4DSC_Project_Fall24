from typing import Any, Callable, Dict, Iterable, List,  Optional, Type

import numpy as np
import math
import wandb
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import config
from eval_config import eval_config
from rewards import get_rewards
from remus100 import step, remus100
from utils import get_obs, next_waypt
import inspect
import csv

class AUVEnv(gym.Env):
    """
    Custom Environment that follows gym interface for AUV adaptive PID controller.
    """
    metadata = {
        "render_modes": ["human"]
        }

    def __init__(
        self,
        render_mode = None,
        config: Dict = config,
        get_rewards: Callable = get_rewards,
        get_obs: Callable = get_obs,
        next_waypt: Callable = next_waypt,
        dynamics_func: Callable = step,
        remus100: Callable = remus100
        
        ):

        super(AUVEnv, self).__init__()

        self.render_mode = render_mode
        self.cfg = config
        self.get_rewards = get_rewards
        self.get_obs = get_obs
        self.next_waypt = next_waypt
        self.dynamics = dynamics_func
        self.remus100 = remus100

        # initialize anything necessary for environment
        self.num_term = 0
        self.num_trunc = 0
        self.simData = np.empty( [0, 23], float)
        self.t = 0
        self.sampleTime = self.cfg["sim_dt"]
        self.max_time = self.cfg["sim_max_time"]
        self.path_list = self.cfg["path_list"]
        self.path = self.path_list[1]
        self.path_idx = 0
        self.ref_rpm = 900
        self.Vc = 0
        init_heading = np.tan(self.path[self.path_idx][0]/max([self.path[self.path_idx][1],0.0001])) / np.pi * 180
        self.vehicle = self.remus100(r_z = self.path[self.path_idx][2], r_psi = init_heading, r_rpm = self.ref_rpm, V_c0 = self.Vc, target_positions = self.path)

        self.waypt_dist = 50; self.final_pt = False
        self.dist_within = self.cfg["waypt_dist_within_criteria"]
        self.dist_limit = self.cfg["waypt_overshoot_criteria"]
        self.depth_err = []; self.yaw_err = []; self.pitch_err = []

        self.eta = self.vehicle.eta
        self.nu = self.vehicle.nu
        self.u_actual = self.vehicle.u_actual

        self.beta = self.cfg["reward_beta_coefficient"]
        self.roll_lim = self.cfg["roll_trunc_limit"]
        self.pitch_lim = self.cfg["pitch_trunc_limit"]

        self.kp_high = self.cfg["Kp_limit"]
        self.ki_high = self.cfg["Ki_limit"]

        # Define action and observation space
        # action space is limits on PID coefficients - Kp, Ki for depth, yaw, pitch
        # controllers respectively
        # self.action_space = spaces.Box(
        #     low=np.array([0, 0, 0, 0, 0, 0]),
        #     high=np.array([self.kp_high, self.ki_high, self.kp_high, self.ki_high, self.kp_high, self.ki_high]),
        #     dtype=np.float32
        #     )
        
        # Normalized action space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1]),
            high=np.array([1,1,1,1,1,1]),
            dtype=np.float32
            )        

        # The observation space is the state vector - [x y z Theta nu del_r del_s
        # controller_errors] where Theta consists of sin and cos of roll, pitch
        # and yaw angles respectively, del_r/s are rudder and stern plane angles,
        # and controller_errors are depth, yaw, and pitch errors respectively
        phi_lim = self.cfg["phi_lim"]
        theta_lim = self.cfg["theta_lim"]
        psi_lim = self.cfg["psi_lim"]
        del_s_lim = self.cfg["stern_plane_limit"]
        del_r_lim = self.cfg["rudder_lim"]
        x_low = -self.cfg["xlim"]; x_high = self.cfg["xlim"]
        y_low = -self.cfg["ylim"]; y_high = self.cfg["ylim"]
        z_low = -self.cfg["zlim"]; z_high = self.cfg["zlim"]
        vel_lim = self.cfg["vel_lim"]
        ang_vel_lim = self.cfg["ang_vel_lim"]
        yaw_err_lim = self.cfg["yaw_err_lim"]
        pitch_err_lim = self.cfg["pitch_err_lim"]
        depth_err_lim = self.cfg["depth_err_lim"]


        min_Theta = [math.sin(-phi_lim),math.cos(-phi_lim),math.sin(-theta_lim),
                     math.cos(-theta_lim),-1.0,-1.0]
        max_Theta = [math.sin(phi_lim),1.0,math.sin(theta_lim),
                     1.0,1.0,1.0]
        min_nu = [-vel_lim, -vel_lim, -vel_lim, -ang_vel_lim, -ang_vel_lim, -ang_vel_lim]
        max_nu = [vel_lim, vel_lim, vel_lim, ang_vel_lim, ang_vel_lim, ang_vel_lim]

        low_vec = ([x_low, y_low, z_low] + min_Theta + min_nu + [-del_r_lim, -del_s_lim]
                   + [-depth_err_lim, -yaw_err_lim, -pitch_err_lim])
        high_vec = ([x_high, y_high, z_high] + max_Theta + max_nu + [del_r_lim, del_s_lim]
                   + [depth_err_lim, yaw_err_lim, pitch_err_lim])

        self.observation_space = spaces.Box(low=np.array(low_vec), high=
                                            np.array(high_vec), dtype=np.float32)
        
        # Initialize plotting variables for tracking
        self.plot = None
        self.counter = 0

        # Reward sum for each timestep, total episode reward
        self.reward_sum = 0
        self.episode_reward = 0
        self.depth_e = 0
        self.yaw_e = 0
        self.pitch_e = 0
        self.rud_act = 0
        self.stern_act = 0
        self.depth_e_ep = 0
        self.yaw_e_ep = 0
        self.pitch_e_ep = 0
        self.rud_act_ep = 0
        self.stern_act_ep = 0

    def reset(self, seed=None, options=None):
        # called to initiate a new episode, called before step function
        # also called whenever terminated or truncated is issued
        # resets environment to an initial state

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Render the data for episode about to be reset and reset counter
        # if self.counter > 1:
        #     if(not 'evaluate' in inspect.stack()[-1][1]):
        #         wandb.log({"plot": wandb.Image(self.render())})
        #         plt.close()
            
        self.counter = 0

        # Reset time counter
        self.t = 0

        # Track cumulative episode rewards in wandb and reset reward sum values
        self.episode_reward = self.reward_sum
        self.depth_e_ep = self.depth_e
        self.yaw_e_ep =  self.yaw_e
        self.pitch_e_ep = self.pitch_e
        self.rud_act_ep = self.rud_act
        self.stern_act_ep = self.stern_act

        self.reward_sum = 0
        self.depth_e = 0
        self.yaw_e = 0
        self.pitch_e = 0
        self.rud_act = 0
        self.stern_act = 0

        # choose path from list of paths in config file, reset path values
        if('evaluate' in inspect.stack()[-1][1]):
            self.path = eval_config["path"]
        else:
            path_choice = self.np_random.integers(1,len(list(self.path_list.values())),endpoint=True)
            self.path = self.path_list[path_choice]
        self.path_idx = 0

        # choose reference propeller rpm in range of values in config file, or set constant
        # self.ref_rpm = self.np_random.uniform(Low=self.cfg["ref_rmp_low"], high=self.cfg["ref_rmp_high"])
        self.ref_rpm = self.cfg["ref_rpm_const"]

        # randomly sample water current velocity over range in config file
        if('evaluate' in inspect.stack()[-1][1]):
            self.Vc = eval_config["Vc"]
        else:
            self.Vc = self.np_random.uniform(low=self.cfg["water_curr_vel_low"], high=self.cfg["water_curr_vel_high"])

        # instantiate remus100 - inputs are initial reference depth (depth of 1st target pt),
        # and reference propeller rpm
        init_heading = np.tan(self.path[self.path_idx][0]/max([self.path[self.path_idx][1],0.0001])) / np.pi * 180
        self.vehicle = self.remus100(r_z = self.path[self.path_idx][2], r_psi = init_heading, r_rpm = self.ref_rpm, V_c0 = self.Vc, target_positions = self.path)

        # set dynamic lift coefficients and vehicle drag coefficient
        if('evaluate' in inspect.stack()[-1][1]):
            self.vehicle.CL_delta_r = eval_config["CL_delta_r"]
            self.vehicle.CL_delta_s = eval_config["CL_delta_s"]
            Cd = eval_config["Cd"]
        else:
            self.vehicle.CL_delta_r = self.np_random.uniform(low=self.cfg["rudder_CL_low"], high=self.cfg["rudder_CL_high"])
            self.vehicle.CL_delta_s = self.np_random.uniform(low=self.cfg["stern_CL_low"], high=self.cfg["stern_CL_high"])
            Cd = self.np_random.uniform(low=self.cfg["Cd_low"], high=self.cfg["Cd_high"])
        self.vehicle.CD_0 = Cd * math.pi * (self.vehicle.diam/2)**2 / self.vehicle.S

        # reset other values
        self.waypt_dist = 50; self.final_pt = False
        self.depth_err = []; self.yaw_err = []; self.pitch_err = []

        # set initital state of AUV
        self.eta = self.vehicle.eta
        self.nu = self.vehicle.nu
        self.u_actual = self.vehicle.u_actual

        # Store simulation data in simData
        signals = (np.array(list(self.eta) + list(self.nu) + [0, 0, 0] + 
                            list(self.u_actual) + self.vehicle.target_position + [self.vehicle.psi_d] + [self.vehicle.theta_d]))
        self.simData = signals
        

        # determine observation from initial state
        z_e = (self.eta[2]-self.vehicle.z_d)
        psi_e = (self.vehicle.psi_d - self.eta[5] + np.pi) % (np.pi*2) - np.pi
        theta_e = 0 #vehicle is initialized at first target point depth
        observation = self.get_obs(self.eta, self.nu, self.u_actual, z_e,
                                   psi_e, theta_e)

        # determine info to return
        info = {}

        # if self.render_mode == "human":
        #   self.render()

        return observation, info


    def step(self, action):
        # using agent actions, run one timestep of the environment's dynamics
        # [self.z_kp, self.z_ki,self.yaw_kp, self.yaw_ki, self.theta_kp, self.theta_ki] = action

        # handling normalized action space
        def kp_norm(norm_action):
            kp = (norm_action + 1)/2 * self.kp_high
            return kp
        
        def ki_norm(norm_action):
            ki = (norm_action + 1)/2 * self.ki_high
            return ki
        
        self.z_kp = kp_norm(action[0])
        self.z_ki = ki_norm(action[1])
        self.yaw_kp = kp_norm(action[2])
        self.yaw_ki = ki_norm(action[3])
        self.theta_kp = kp_norm(action[4])
        self.theta_ki = ki_norm(action[5])


        self.t += self.sampleTime

        # determine updated observation based on action
        [self.vehicle, self.eta, self.nu, self.sampleTime, self.u_actual, self.simData] = (
            self.dynamics(self.vehicle, self.eta, self.nu, self.sampleTime,
                              self.u_actual, self.simData)
        )

        observation = self.get_obs(self.eta, self.nu, self.u_actual, self.vehicle.z_previous_error,
                                   self.vehicle.yaw_previous_error, self.vehicle.theta_previous_error)

        # update target waypoint based on current state
        [self.waypt_dist, self.final_pt, self.path_idx] = (
            self.next_waypt(self.path, self.path_idx, self.vehicle, self.dist_within)
        )

        # calculate reward, log on wandb
        self.depth_err.append(self.vehicle.z_previous_error)
        self.yaw_err.append(self.vehicle.yaw_previous_error)
        self.pitch_err.append(self.vehicle.theta_previous_error)
        [reward,indiv_rew_terms] = self.get_rewards(self.vehicle, self.depth_err, self.yaw_err, self.pitch_err,
                             self.simData, self.beta)
        
        self.reward_sum += reward
        self.depth_e += indiv_rew_terms[0]
        self.yaw_e += indiv_rew_terms[1]
        self.pitch_e += indiv_rew_terms[2]
        self.rud_act += indiv_rew_terms[3]
        self.stern_act += indiv_rew_terms[4]

        if(not 'evaluate' in inspect.stack()[-1][1]):
            wandb.log({"train/reward":self.reward_sum})
            wandb.log({"train/episode_return":self.episode_reward})
            wandb.log({"reward/depth_err":self.depth_e_ep})
            wandb.log({"reward/yaw_err":self.yaw_e_ep})
            wandb.log({"reward/pitch_err":self.pitch_e_ep})
            wandb.log({"reward/rudder_act":self.rud_act_ep})
            wandb.log({"reward/stern_plane_act":self.stern_act_ep})

        # set terminated criteria - if reached final waypt
        terminated = self.final_pt

        # set truncated criteria
        # if over max time, if dist to next waypt is over limit, if roll/pitch over 30/40deg respectively
        truncated = False
        if ((self.t > self.max_time) or (self.waypt_dist > self.dist_limit) or
                     (abs(self.eta[3]) > self.roll_lim) or (abs(self.eta[4]) > self.pitch_lim)):
           truncated = True

        # set value based on terminated or truncated for episode 
        if terminated:
            self.num_term += 1

        if truncated:
            self.num_trunc += 1  

        # determine info to return
        info = {}

        # Iterate plotting counter and log plot if step is divisible by 100
        self.counter += 1
        
        if(not 'evaluate' in inspect.stack()[-1][1]):
            wandb.log({"state/rudder_angle":self.u_actual[0]})
            wandb.log({"state/stern_plane_angle":self.u_actual[1]})
            wandb.log({"state/depth_error":self.vehicle.z_previous_error})
            wandb.log({"state/yaw_error":self.vehicle.yaw_previous_error})
            wandb.log({"state/pitch_error":self.vehicle.theta_previous_error})
            wandb.log({"state/depth_Kp":self.z_kp})
            wandb.log({"state/depth_Ki":self.z_ki})
            wandb.log({"state/yaw_Kp":self.yaw_kp})
            wandb.log({"state/yaw_Ki":self.yaw_ki})
            wandb.log({"state/pitch_Kp":self.theta_kp})
            wandb.log({"state/pitch_Ki":self.theta_ki})

            
            # log number of episodes that are terminated vs truncated
            wandb.log({"reward/num_term":self.num_term})
            wandb.log({"reward/num_trunc":self.num_trunc})


        if('evaluate' in inspect.stack()[-1][1]):
            with open('PPO_AUV_eval.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.simData[-1,:])

        return observation, reward, terminated, truncated, info


    def render(self):
      figSize1 = [25, 13]  # figure1 size in cm
      dpiValue = 150  # figure dpi value

      def cm2inch(value):  # inch to cm
          return value / 2.54

      # State vectors
      x = self.simData[:,0]
      y = self.simData[:,1]
      z = self.simData[:,2]

      # down-sampling the xyz data points
      N = y
      E = x
      D = -z  # Down is negative z

      # Attaching 3D axis to the figure
      fig = plt.figure(figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
          dpi=dpiValue)
      ax = fig.add_subplot(projection = '3d')

      # Setting the axes properties
      ax.set_xlabel('X / East')
      ax.set_ylabel('Y / North')
      ax.set_zlim3d([-100, 5])                   # default depth = -100 m

      if np.amax(z) > 100.0:
          ax.set_zlim3d([-np.amax(z), 20])

      ax.set_zlabel('-Z / Down')

      # Plot trajectory
      ax.plot(E, N, D)

      # Plot 2D surface for z = 0
      [x_min, x_max] = ax.get_xlim()
      [y_min, y_max] = ax.get_ylim()
      x_grid = np.arange(x_min-20, x_max+20)
      y_grid = np.arange(y_min-20, y_max+20)
      [xx, yy] = np.meshgrid(x_grid, y_grid)
      zz = 0 * xx
      ax.plot_surface(xx, yy, zz, alpha=0.3, color='b')

      # Title of plot
      ax.set_title('North-East-Down')

      return fig


    def close(self):
      return
    

# Perform environment checks
if __name__ == "__main__":
    import time
    from stable_baselines3.common.env_checker import check_env

    # SB3 environment check
    # If the environment don't follow the correct interface, an error will be thrown
    env = AUVEnv()
    
    check_env(env, warn=True)

    # test environment
    obs = env.reset()
    # env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    T = 100
    now = time.time()
    for _ in range(T):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
    print(f"{int(T/(time.time() - now)):_d} steps/second")
    env.render()
    plt.show()
