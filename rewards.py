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
    del_delta_h = simData[-1][13] - simData[-2][13]
    del_delta_v = simData[-1][12] - simData[-2][12]
    delta_h_max = vehicle.deltaMax_s
    delta_v_max = vehicle.deltaMax_r
    e_depth_min = np.min(np.absolute(depth_err))
    e_yaw_min = np.min(np.absolute(yaw_err))
    e_pitch_min = np.min(np.absolute(pitch_err))

    # define functions for T operator of reward function
    def T_op(err, err_min):
        # if abs(err) <= err_min:
        if abs(abs(err) - abs(err_min)) <= 0.001:
            T = math.exp(-abs(err))
        else:
            # T = -abs(err) / math.pi
            T = 0
        return T
    
    def T_op_d(err, err_min):
        # if abs(err) <= err_min:
        if abs(abs(err) - abs(err_min)) <= 0.001:
            T = math.exp(-abs(err))
        else:
            # T = -abs(err) / cfg["depth_err_lim"]
            T = 0
        return T
    
    T_depth = T_op_d(e_depth, e_depth_min)
    T_yaw = T_op(e_yaw, e_yaw_min)
    T_pitch = T_op(e_pitch, e_pitch_min)

    # Penalizing rudder commands by difference in last two commands as fraction of maximum rudder command
    # neg_rud_act = beta*(del_delta_v / delta_v_max)
    # neg_stern_plane_act = beta*(del_delta_h / delta_h_max)

    # First initialize rudder command penalties
    neg_rud_act = 0
    neg_stern_plane_act = 0

    # penalizing rudder commands over threshold of +/- 25deg
    delta_h = simData[-1][13]
    delta_v = simData[-1][12]

    if abs(delta_v) >= math.radians(25):
        diff_v = abs(abs(delta_v) - math.radians(25))
        diff_v_deg = math.degrees(diff_v)
        neg_rud_act += beta * (math.exp(-1/diff_v_deg))
    
    if abs(delta_h) >= math.radians(25):
        diff_h = abs(abs(delta_h) - math.radians(25))
        diff_h_deg = math.degrees(diff_h)
        neg_stern_plane_act += beta * (math.exp(-1/diff_h_deg))


    # Penalizing if greatest difference in last 5 rudder commands is over threshold +/- 15deg
    rud_cmd_lst = simData[:,12]
    splane_cmd_lst = simData[:,13]
    lg_r = len(rud_cmd_lst)
    lg_s = len(splane_cmd_lst)
    if lg_r < 6:
        end_r = rud_cmd_lst
        end_s = splane_cmd_lst
    else:
        end_r = rud_cmd_lst[lg_r-5:lg_r]
        end_s = splane_cmd_lst[lg_s-5:lg_s]

    diff_r = abs(max(end_r) - min(end_r))
    diff_s = abs(max(end_s) - min(end_s))

    if diff_r >= math.radians(15):
        neg_rud_act += beta * math.exp(-1/math.degrees(diff_r))

    if diff_s >= math.radians(15):
        neg_stern_plane_act += beta * math.exp(-1/math.degrees(diff_s))


    # Scale down yaw error term by alpha coefficient
    T_yaw = cfg["reward_alpha_coefficient"] * T_yaw

    # Scale up pitch error reward and actuation penalty
    # T_pitch = 1.5 * T_pitch
    # neg_stern_plane_act = 2 * neg_stern_plane_act

    # Output individual reward terms as well as sum
    indiv_terms = [T_depth,T_yaw,T_pitch,neg_rud_act,neg_stern_plane_act]

    r_t = (T_depth + T_yaw + T_pitch + neg_rud_act + neg_stern_plane_act)

    return r_t, indiv_terms


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