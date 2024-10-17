import math
import numpy as np
from config import config as cfg

# Rewards function
def get_rewards(vehicle, depth_err, yaw_err, pitch_err, simData, beta):
    """
    r_t = get_rewards(vehicle, depth_err, yaw_err, pitch_err, simData, beta) returns the RL reward value
    using arrays of current and all past angular errors, and arrays of all horizontal
    and vertical rudder angle commands as well as a coefficient beta to adjust weights
    """

    # break out values needed from input arrays
    e_depth = depth_err[-1]; e_yaw = yaw_err[-1]; e_pitch = pitch_err[-1]
    del_delta_v = simData[-1][13] - simData[-2][13]
    del_delta_h = simData[-1][12] - simData[-2][12]
    delta_v_max = vehicle.deltaMax_s
    delta_h_max = vehicle.deltaMax_r
    e_depth_min = np.min(np.absolute(depth_err))
    e_yaw_min = np.min(np.absolute(yaw_err))
    e_pitch_min = np.min(np.absolute(pitch_err))

    # define functions for T operator of reward function
    def T_op(err, err_min):
        if abs(err) <= err_min:
            T = math.exp(-abs(err))
        else:
            T = -abs(err) / math.pi
        return T
    
    def T_op_d(err, err_min):
        if abs(err) <= err_min:
            T = math.exp(-abs(err))
        else:
            T = -abs(err) / cfg["depth_err_lim"]
        return T

    r_t = (T_op_d(e_depth, e_depth_min) + T_op(e_yaw, e_yaw_min) + T_op(e_pitch, e_pitch_min)
        + beta*(del_delta_v / delta_v_max) + beta*(del_delta_h / delta_h_max))

    return r_t


# test the function
if __name__ == "__main__":

    #from remus100 import remus100
    from remus100 import remus100

    vehicle = remus100(30,0,900)
    depth_err = [10,2,5]
    yaw_err = [3,1.5,0.75]
    pitch_err = [0.75,0.375,0.125]
    s1 = np.array([0,0,5,0,0,0,2,0,0,0,0,0,0,0,900,0,0,900,0,100,30,0.75])
    s2 = np.array([0,1,6,0,0,0.4,2,0,0,0,0,0,0.4,0.4,900,0.2,0.2,900,0,100,30,0.75])
    simData = np.vstack([s1,s2])
    beta = -0.01

    r_t = get_rewards(vehicle, depth_err, yaw_err, pitch_err, simData, beta)

    print(r_t)