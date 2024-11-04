import torch
import math

config = {
    "num_envs": 1,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.2,
    "policy_dt": 0.05,

    "sim_max_time": 1800,
    "max_steps": 1_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 50,
    "minibatch_size": 50,
    "model_save_freq": 1000,

    # action/observation space limits
    "phi_lim": math.radians(30),
    "theta_lim": math.radians(45),
    "psi_lim": math.radians(180),
    "stern_plane_limit": math.radians(30),
    "rudder_lim": math.radians(30),
    "xlim": 500,
    "ylim": 500,
    "zlim": 100,
    "vel_lim": 10,
    "ang_vel_lim": math.pi,
    "yaw_err_lim": math.pi,
    "pitch_err_lim": math.pi,
    "depth_err_lim": 100,

    "Kp_limit": 10,
    "Ki_limit": 1,

    # path configuration values
    "path_list": {
        1: [[0,100,30],[100,100,30],[100,0,20],[0,0,20]],
        2: [[0,60,10],[0,120,15],[30,190,15],[100,220,15],[170,190,20],[200,120,25]],
        3: [[30,25,5],[40,100,10],[40,150,10],[20,160,10],[0,125,10],[0,50,10],[0,0,15],[30,-5,20]],
        4: [[30,0,8],[60,0,10],[30,100,13],[0,200,15],[30,300,18],[60,400,20],[30,500,22]],
        5: [[60,0,5],[80,100,10],[60,150,10],[0,200,10],[-20,250,10],[-40,300,10],[-20,350,10],[20,400,15]]
    },
    "waypt_dist_within_criteria": 5,
    "waypt_overshoot_criteria": 200,

    # reward and truncate limits
    "reward_beta_coefficient": -0.0001,
    "roll_trunc_limit": math.radians(30),
    "pitch_trunc_limit": math.radians(40),

    # reset limits
    "ref_rpm_const": 900.0,
    "ref_rpm_low": 600.0,
    "ref_rpm_high": 1800.0,
    "rudder_CL_low": 2.0,
    "rudder_CL_high": 4.0,
    "stern_CL_low": 2.0,
    "stern_CL_high": 4.0,
    "Cd_low": 2.0,
    "Cd_high": 4.0,
    "water_curr_vel_low": 0,
    "water_curr_vel_high": 0.5
}