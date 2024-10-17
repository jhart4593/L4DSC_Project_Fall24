import numpy as np
import math


# Function to calculate environment observation from state of AUV
def get_obs(eta, nu, u_actual, depth_e, yaw_e, pitch_e):
  """
  s_t = get_obs(eta, nu, u_actual, depth_e, yaw_e, pitch_e)
  returns the observation array given the current state of the
  AUV.
  """
  sin_ang = np.sin(eta)
  cos_ang = np.cos(eta)
  Theta = [sin_ang[3], cos_ang[3], sin_ang[4], cos_ang[4], sin_ang[5], cos_ang[5]]

  del_r = u_actual[0]
  del_s = u_actual[1]

  s_t = np.array([eta[0], eta[1], eta[2]] + Theta + list(nu) + [del_r, del_s] + [depth_e, yaw_e, pitch_e])

  return np.float32(s_t)


# Waypoint iterator  
def next_waypt(path, path_idx, vehicle, dist_within_waypt):
  """
  [waypt_dist, final_pt, path_idx] = next_waypt(path, path_idx, vehicle, dist_within_waypt)
  checks for distance to next waypoint and iterates on the next waypoint as a target
  if within 'dist_within_waypt' of the current target. It returns the distance to the next waypoint
  as well as False if still traversing the path or True if reached final waypoint.
  """
  final_pt = False
  waypt_dist = np.linalg.norm(np.array(vehicle.target_position) - np.array(vehicle.current_position))
  if waypt_dist < dist_within_waypt:
    path_idx += 1
    if path_idx >= len(path):
        final_pt = True
    else:
      vehicle.previous_target_position = vehicle.target_position
      vehicle.target_position = path[path_idx]
      vehicle.ref_z = path[path_idx][2]

  return waypt_dist, final_pt, path_idx


# test the functions
if __name__ == "__main__":

  from remus100 import remus100

  eta = np.array([1,2,3,0,math.pi/6,math.pi/4])
  nu = np.array([2,1,3,0.2,0.5,0.7])
  u_actual = np.array([math.pi/3,math.pi/5,900])
  depth_e = 5
  yaw_e = 0.3
  pitch_e = 0.4

  st = get_obs(eta,nu,u_actual,depth_e,yaw_e,pitch_e)
  print(st)

  # path = [[0,100,30],[100,100,30],[100,0,20],[0,0,20]]
  # path = [[0,0,4]]
  path = [[0,0,4],[10,20,30]]
  path_idx = 0
  vehicle = remus100(30,0,900,target_positions=path)
  dist_within_waypt = 5

  [waypt_dist, final_pt, path_idx] = next_waypt(path,path_idx,vehicle,dist_within_waypt)
  print(waypt_dist,final_pt, path_idx)
  print(vehicle.target_position,vehicle.current_position)