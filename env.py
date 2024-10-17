from typing import Any, Callable, Dict, Iterable, List,  Optional, Type

import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from config import config
from rewards import get_rewards
from remus100 import step, remus100
from utils import get_obs, next_waypt



class AUVEnv(gym.Env):
    """
    Custom Environment that follows gym interface for AUV adaptive PID controller.
    """
    metadata = {
        "render_modes": ["human"],
        "render_fps": int(1 / config["policy_dt"])
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
        self.simData = np.empty( [0, 22], float)
        self.t = 0
        self.sampleTime = self.cfg["sim_dt"]
        self.max_time = self.cfg["sim_max_time"]
        self.path_list = self.cfg["path_list"]
        self.path = self.path_list[1]
        self.path_idx = 0
        self.ref_rpm = 900
        self.Vc = 0
        self.vehicle = self.remus100(r_rpm=self.ref_rpm)

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
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([self.kp_high, self.ki_high, self.kp_high, self.ki_high, self.kp_high, self.ki_high]),
            dtype=np.float32
            )

        # The observation space is the state vector - [x y z Theta nu del_r del_s
        #  controller_errors] where Theta consists of sin and cos of roll, pitch
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

    def reset(self, seed=None, options=None):
        # called to initiate a new episode, called before step function
        # also called whenever terminated or truncated is issued
        # resets environment to an initial state

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset time counter
        self.t = 0

        # choose path from list of paths in config file, reset path values
        path_choice = self.np_random.integers(1,len(list(self.path_list.values())),endpoint=True)
        self.path = self.path_list[path_choice]
        self.path_idx = 0

        # choose reference propeller rpm in range of values in config file, or set constant
        # self.ref_rpm = self.np_random.uniform(Low=self.cfg["ref_rmp_low"], high=self.cfg["ref_rmp_high"])
        self.ref_rpm = self.cfg["ref_rpm_const"]

        # randomly sample water current velocity over range in config file
        self.Vc = self.np_random.uniform(low=self.cfg["water_curr_vel_low"], high=self.cfg["water_curr_vel_high"])

        # instantiate remus100 - inputs are initial reference depth (depth of 1st target pt),
        # and reference propeller rpm
        self.vehicle = self.remus100(self.path[self.path_idx][2], self.ref_rpm, self.Vc, target_positions = self.path)

        # set dynamic lift coefficients and vehicle drag coefficient
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
                            list(self.u_actual) + self.vehicle.target_position + [self.vehicle.psi_d]))
        self.simData = signals
        

        # determine observation from initial state
        z_e = (self.eta[2]-self.vehicle.z_d)
        psi_e = (self.vehicle.psi_d - self.eta[5] + np.pi) % (np.pi*2) - np.pi
        theta_e = 0 #vehicle is initialized at first target point depth
        observation = self.get_obs(self.eta, self.nu, self.u_actual, z_e,
                                   psi_e, theta_e)

        # determine info to return
        info = {}

        if self.render_mode == "human":
          self.render()

        return observation, info


    def step(self, action):
        # using agent actions, run one timestep of the environment's dynamics
        [self.z_kp, self.z_ki,self.yaw_kp, self.yaw_ki, self.theta_kp, self.theta_ki] = action

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

        # calculate reward
        self.depth_err.append(self.vehicle.z_previous_error)
        self.yaw_err.append(self.vehicle.yaw_previous_error)
        self.pitch_err.append(self.vehicle.theta_previous_error)
        reward = self.get_rewards(self.vehicle, self.depth_err, self.yaw_err, self.pitch_err,
                             self.simData, self.beta)

        # set terminated criteria - if reached final waypt
        terminated = self.final_pt

        # set truncated criteria
        # if over max time, if dist to next waypt is over limit, if roll/pitch over 30/40deg respectively
        truncated = False
        if ((self.t > self.max_time) or (self.waypt_dist > self.dist_limit) or
                     (abs(self.eta[3]) > self.roll_lim) or (abs(self.eta[4]) > self.pitch_lim)):
           truncated = True
           


        # determine info to return
        info = {}

        return observation, reward, terminated, truncated, info


    def render(self):
      legendSize = 10  # legend size
      figSize1 = [25, 13]  # figure1 size in cm
      dpiValue = 150  # figure dpi value

      def R2D(value):  # radians to degrees
          return value * 180 / math.pi

      def cm2inch(value):  # inch to cm
          return value / 2.54

      # State vectors
      x = self.simData[:,0]
      y = self.simData[:,1]
      z = self.simData[:,2]

      # down-sampling the xyz data points
      N = y[::len(x)]
      E = x[::len(x)]
      D = z[::len(x)]

      dataSet = np.array([N, E, -D])      # Down is negative z

      # Attaching 3D axis to the figure
      fig = plt.figure(figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
                dpi=dpiValue)
      ax = p3.Axes3D(fig, auto_add_to_figure=False)
      fig.add_axes(ax)

      # Line/trajectory plot
      line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b')[0]

      # Setting the axes properties
      ax.set_xlabel('X / East')
      ax.set_ylabel('Y / North')
      ax.set_zlim3d([-100, 5])                   # default depth = -100 m

      if np.amax(z) > 100.0:
          ax.set_zlim3d([-np.amax(z), 20])

      ax.set_zlabel('-Z / Down')

      # Plot 2D surface for z = 0
      [x_min, x_max] = ax.get_xlim()
      [y_min, y_max] = ax.get_ylim()
      x_grid = np.arange(x_min-20, x_max+20)
      y_grid = np.arange(y_min-20, y_max+20)
      [xx, yy] = np.meshgrid(x_grid, y_grid)
      zz = 0 * xx
      ax.plot_surface(xx, yy, zz, alpha=0.3)

      # Title of plot
      ax.set_title('North-East-Down')


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

    T = 10
    now = time.time()
    for _ in range(T):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
    print(f"{int(T/(time.time() - now)):_d} steps/second")
    #env.render()