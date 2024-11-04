#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remus100.py:  

   Class for the Remus 100 cylinder-shaped autonomous underwater vehicle (AUV), 
   which is controlled using a tail rudder, stern planes and a propeller. The 
   length of the AUV is 1.6 m, the cylinder diameter is 19 cm and the 
   mass of the vehicle is 31.9 kg. The maximum speed of 2.5 m/s is obtained 
   when the propeller runs at 1525 rpm in zero currents.
       
   remus100()                           
       Step input, stern plane, rudder and propeller revolution     
   
    remus100(z_d,psi_d,n_d,V_c,target_positions)
        z_d:    desired depth (m), positive downwards
        psi_d:  desired yaw angle (deg)
        n_d:    desired propeller revolution (rpm)
        V_c0:    current speed (m/s)              

Methods:
        
    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime ) returns 
        nu[k+1] and u_actual[k+1] using Euler's method. The control input is:

            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]

    u = depthHeadingAutopilot(eta,nu,sampleTime) 
        Simultaneously control of depth and heading using controllers of 
        PID and SMC ype. Propeller rpm is given as a step command.
       
    u = stepInput(t) generates tail rudder, stern planes and RPM step inputs.   
       
References: 
    
    B. Allen, W. S. Vorus and T. Prestero, "Propulsion system performance 
         enhancements on REMUS AUVs," OCEANS 2000 MTS/IEEE Conference and 
         Exhibition. Conference Proceedings, 2000, pp. 1869-1873 vol.3, 
         doi: 10.1109/OCEANS.2000.882209.    
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
         Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Main program for the Python Vehicle Simulator, which can be used
    to simulate and test guidance, navigation and control (GNC) systems.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd edition, John Wiley & Sons, Chichester, UK. 
URL: https://www.fossen.biz/wiley  
    
Author:     Thor I. Fossen
"""

###############################################################################
# Main simulation loop 
###############################################################################
def main():    
    # Simulation parameters: 
    sampleTime = 0.02                   # sample time [seconds]
    runTime = 7
    N = int(runTime/sampleTime*60)

    # Set Target Positions
    target_positions = [[0,100,30],[100,100,30],[100,0,20],[0,0,20]]

    # Initialize Vehicle
    vehicle = remus100(30,0,900,N,0.2,target_positions)
    
    # Initial state vectors
    eta = vehicle.eta   # position/attitude, defined by vehicle class
    nu = vehicle.nu                              # velocity, defined by vehicle class
    u_actual = vehicle.u_actual                  # actual inputs, defined by vehicle class
    
    # Initialization of table used to store the simulation data
    j = 0
    simData = np.empty( [0, 22], float)
    simTime = []

    # Simulator for-loop
    for i in range(0,N+1):
        
        t = i * sampleTime
        simTime.append(t)

        # Step Function
        vehicle,eta,nu,sampleTime,u_actual,simData = step(vehicle,eta,nu,sampleTime,u_actual,simData)

        if np.linalg.norm(np.array(vehicle.target_position) - np.array(vehicle.current_position)) < 5:
            j += 1
            if j >= len(target_positions):
                break
            vehicle.previous_target_position = vehicle.target_position
            vehicle.target_position = target_positions[j]
            vehicle.ref_z = target_positions[j][2]

    plotVehicleStates(simTime, simData, 'vehicle_states.png', 1)                    
    plotControls(simTime, simData, 'vehicle_controls.png', 2)
    plot3D(simData, target_positions, 50, 10, 'vehicle_3D.gif', 3)   
    
    plt.show()
    plt.close()

def step(vehicle,eta,nu,sampleTime,u_actual,simData):
    # Vehicle specific control systems
    u_control = vehicle.depthHeadingAutopilot(eta,nu,sampleTime)             
    
    # Store simulation data in simData
    signals = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + vehicle.target_position + [vehicle.psi_d])
    simData = np.vstack( [simData, signals] ) 

    # Water Current Model
    vehicle.water_currents()

    # Propagate vehicle and attitude dynamics
    [nu, u_actual, _]  = vehicle.dynamics(eta,nu,u_actual,u_control,sampleTime)
    eta = vehicle.attitudeEuler(eta,nu,sampleTime)
    [nu, u_actual] = vehicle.addnoise(nu, u_actual)

    return vehicle,eta,nu,sampleTime,u_actual,simData

# Class Vehicle
class remus100:
    """
    remus100()
        Rudder angle, stern plane and propeller revolution step inputs
        
    remus100('depthHeadingAutopilot',z_d,psi_d,n_d,N,V_c,target_positions) 
        Depth and heading autopilots
        
    Inputs:
        z_d:    desired depth, positive downwards (m)
        psi_d:  desired heading angle (deg)
        n_d:    desired propeller revolution (rpm)
        N:      iterations
        V_c:    current speed (m/s)
        target_positions: [x,y,z] target positions
    """

    def __init__(
        self,
        r_z = 0,
        r_psi = 0,
        r_rpm = 0,
        N = 0,
        V_c0 = 0,
        target_positions = [[0,0,10]]
    ):

        # Constants
        self.D2R = math.pi / 180        # deg2rad
        self.rho = 1026                 # density of water (kg/m^3)
        g = 9.81                        # acceleration of gravity (m/s^2)
            
        self.ref_z = target_positions[0][2]
        self.ref_psi = r_psi
        self.ref_n = r_rpm
        self.V_c0 = V_c0
        self.V_c = V_c0
        self.beta_c = 0 * self.D2R
        self.alpha_c = 0 * self.D2R        
        
        # Initialize the AUV model 
        self.L = 1.6                # length (m)
        self.diam = 0.19            # cylinder diameter (m)
             
        # Hydrodynamics (Fossen 2021, Section 8.4.2)    
        self.S = 0.7 * self.L * self.diam    # S = 70% of rectangle L * diam
        a = self.L/2                         # semi-axes
        b = self.diam/2                  
        self.r_bg = np.array([0, 0, 0.02], float)    # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)       # CB w.r.t. to the CO

        # Actuator dynamics
        self.deltaMax_r = 30 * self.D2R # max rudder angle (rad)
        self.deltaMax_s = 30 * self.D2R # max stern plane angle (rad)
        self.nMax = 1525                # max propeller revolution (rpm)    
        self.T_delta = 0.1              # rudder/stern plane time constant (s)
        self.T_n = 0.1                  # propeller time constant (s)

        if r_rpm < 0.0 or r_rpm > self.nMax:
            sys.exit("The RPM value should be in the interval 0-%s", (self.nMax))
        
        if r_z > 100.0 or r_z < 0.0:
            sys.exit('desired depth must be between 0-100 m')    
        
        
        # Hydrodynamics (Fossen 2021, Section 8.4.2)    
        self.S = 0.7 * self.L * self.diam    # S = 70% of rectangle L * diam
        a = self.L/2                         # semi-axes
        b = self.diam/2                  
        self.r_bg = np.array([0, 0, 0.02], float)    # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)       # CB w.r.t. to the CO


        # Hydrodynamics (Fossen 2021, Section 8.4.2)    
        self.S = 0.7 * self.L * self.diam    # S = 70% of rectangle L * diam
        a = self.L/2                         # semi-axes
        b = self.diam/2                  
        self.r_bg = np.array([0, 0, 0.02], float)    # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)       # CB w.r.t. to the CO

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)   
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42                              # from Allen et al. (2000)
        self.CD_0 = Cd * math.pi * b**2 / self.S
        
        # Rigid-body mass matrix expressed in CO
        m = 4/3 * math.pi * self.rho * a * b**2     # mass of spheriod 
        Ix = (2/5) * m * b**2                       # moment of inertia
        Iy = (1/5) * m * (a**2 + b**2)
        Iz = Iy
        MRB_CG = np.diag([ m, m, m, Ix, Iy, Iz ])   # MRB expressed in the CG     
        H_rg = self.Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg           # MRB expressed in the CO

        # Weight and buoyancy
        self.W = m * g
        self.B = self.W
        
        # Added moment of inertia in roll: A44 = r44 * Ix
        r44 = 0.3           
        MA_44 = r44 * Ix
        
        # Lamb's k-factors
        e = math.sqrt( 1-(b/a)**2 )
        alpha_0 = ( 2 * (1-e**2)/pow(e,3) ) * ( 0.5 * math.log( (1+e)/(1-e) ) - e )  
        beta_0  = 1/(e**2) - (1-e**2) / (2*pow(e,3)) * math.log( (1+e)/(1-e) )

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0  / (2 - beta_0)
        k_prime = pow(e,4) * (beta_0-alpha_0) / ( 
            (2-e**2) * ( 2*e**2 - (2-e**2) * (beta_0-alpha_0) ) )   

        # Added mass system matrix expressed in the CO
        self.MA = np.diag([ m*k1, m*k2, m*k2, MA_44, k_prime*Iy, k_prime*Iy ])
          
        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Natural frequencies in roll and pitch
        self.w_roll = math.sqrt( self.W * ( self.r_bg[2]-self.r_bb[2] ) / 
            self.M[3][3] )
        self.w_pitch = math.sqrt( self.W * ( self.r_bg[2]-self.r_bb[2] ) / 
            self.M[4][4] )
            
        S_fin = 0.00665             # fin area
        
        # Tail rudder parameters
        self.CL_delta_r = 3.0       # rudder lift coefficient
        self.A_r = 2 * S_fin        # rudder area (m2)
        self.x_r = -a               # rudder x-position (m)

        # Stern-plane parameters (double)
        self.CL_delta_s = 3.0       # stern-plane lift coefficient
        self.A_s = 2 * S_fin        # stern-plane area (m2)
        self.x_s = -a               # stern-plane z-position (m)

        # Low-speed linear damping matrix parameters
        self.T_surge = 20           # time constant in surge (s)
        self.T_sway = 20            # time constant in sway (s)
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3        # relative damping ratio in roll
        self.zeta_pitch = 0.8       # relative damping ratio in pitch
        self.T_yaw = 1              # time constant in yaw (s)
        
        self.e_psi_int = 0     # yaw angle error integral state
        
        self.yaw_kp = 0.5
        self.yaw_ki = 0.001

        self.wn_d_z = 0.05     # desired natural frequency, reference model

        self.z_kp = 1.0        # heave proportional gain, outer loop
        self.z_ki = 0.001
        self.theta_kp = 0.1   # pitch PID controller     
        self.theta_ki = 0.001
        
        self.e_z_int = 0         # heave position integral state
        self.z_d = r_z         # desired position, LP filter initial state
        self.e_theta_int = 0     # pitch angle integral state 

        trim_final, iterations, error = self.trim()

        # calculate (u,v,w)
        self.eta = np.array([0, 0, r_z, trim_final.get('phi_t'), 0, self.ref_psi * self.D2R])
        self.nu = np.array([trim_final.get('u_t'), trim_final.get('v_t'), trim_final.get('w_t'), 0, 0, 0], float) # velocity vector
        self.u_actual = np.array([trim_final.get('ds_t') * self.D2R, trim_final.get('dr_t') * self.D2R, self.ref_n], float)    # control input vector
        
        self.controls = [
            "Tail rudder (deg)",
            "Stern plane (deg)",
            "Propeller revolution (rpm)"
            ]
        self.dimU = len(self.controls) 

        self.nu_noise = np.random.normal(0.0, 0.0025, [self.nu.shape[0],N+1])
        self.u_actual_noise = np.random.normal(0.0, 0.0025, [self.u_actual.shape[0],N+1])
        self.eta_p_noise = np.random.normal(0.0, 0.25, [3,N+1])
        self.eta_v_noise = np.random.normal(0.0, 0.00025, [3,N+1])
        self.V_c_noise = np.random.normal(0.05, 0.0025, [1,N+1])
        self.beta_c_noise = np.random.normal(0.0, 0.25, [1,N+1])       
        self.alpha_c_noise = np.random.normal(0.0, 0.00025, [1,N+1])   

        self.target_position = target_positions[0]
        self.previous_target_position = [0, 0, 0]
        self.current_position = [0, 0, 0]
        self.psi_d = self.heading_control(self.nu)
        self.eta[5] = self.psi_d

        self.noise_counter = 0   

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the AUV equations of motion using Euler's method.
        """

        # Current velocities
        u_c = self.V_c * math.cos(self.alpha_c - eta[4]) * math.cos(self.beta_c - eta[5])  # current surge velocity
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway velocity
        w_c = self.V_c * math.sin(self.alpha_c - eta[4]) * math.cos(self.beta_c - eta[5]) 

        nu_c = np.array([u_c, v_c, w_c, 0, 0, 0], float) # current velocity 
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = nu - nu_c                               # relative velocity    
        alpha = math.atan2( nu_r[2], nu_r[0] )         # angle of attack 
        U = math.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)  # vehicle speed
        U_r = math.sqrt(nu_r[0]**2 + nu_r[1]**2 + nu_r[2]**2)  # relative speed

        # Commands and actual control signals
        delta_r_c = u_control[0]    # commanded tail rudder (rad)
        delta_s_c = u_control[1]    # commanded stern plane (rad)
        n_c = u_control[2]          # commanded propeller revolution (rpm)
        
        delta_r = u_actual[0]       # actual tail rudder (rad)
        delta_s = u_actual[1]       # actual stern plane (rad)
        n = u_actual[2]             # actual propeller revolution (rpm)
        
        # Amplitude saturation of the control signals
        if abs(delta_r) >= self.deltaMax_r:
            delta_r = np.sign(delta_r) * self.deltaMax_r
            
        if abs(delta_s) >= self.deltaMax_s:
            delta_s = np.sign(delta_s) * self.deltaMax_s          
            
        if abs(n) >= self.nMax:
            n = np.sign(n) * self.nMax       
        
        # Propeller coeffs. KT and KQ are computed as a function of advance no.
        # Ja = Va/(n*D_prop) where Va = (1-w)*U = 0.944 * U; Allen et al. (2000)
        D_prop = 0.14   # propeller diameter corresponding to 5.5 inches
        t_prop = 0.1    # thrust deduction number
        n_rps = n / 60  # propeller revolution (rps) 
        Va = 0.944 * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        Ja_max = 0.6632
        
        # Single-screw propeller with 3 blades and blade-area ratio = 0.718.
        # Coffes. are computed using the Matlab MSS toolbox:     
        # >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
        KT_0 = 0.4566
        KQ_0 = 0.0700
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3) 
        KT_max = 0.1798
        KQ_max = 0.0312
        
        # Propeller thrust and propeller-induced roll moment
        # Linear approximations for positive Ja values
        # KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
        # KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja  
      
        if n_rps > 0:   # forward thrust

            X_prop = self.rho * pow(D_prop,4) * ( 
                KT_0 * abs(n_rps) * n_rps + (KT_max-KT_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )        
            K_prop = self.rho * pow(D_prop,5) * (
                KQ_0 * abs(n_rps) * n_rps + (KQ_max-KQ_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )           
            
        else:    # reverse thrust (braking)
        
            X_prop = self.rho * pow(D_prop,4) * KT_0 * abs(n_rps) * n_rps 
            K_prop = self.rho * pow(D_prop,5) * KQ_0 * abs(n_rps) * n_rps 
        
        # Rigi-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = self.m2c(self.MRB, nu_r)
        CA  = self.m2c(self.MA, nu_r)
        
        # CA-terms in roll, pitch and yaw can destabilize the model if quadratic
        # rotational damping is missing. These terms are assumed to be zero
        CA[4][0] = 0     # Quadratic velocity terms due to pitching
        CA[0][4] = 0    
        CA[4][2] = 0
        CA[2][4] = 0
        CA[5][0] = 0     # Munk moment in yaw 
        CA[0][5] = 0
        CA[5][1] = 0
        CA[1][5] = 0
        
        C = CRB + CA

        # Dissipative forces and moments
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll  * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw
            ])
        
        # Linear surge and sway damping
        D[0][0] = D[0][0] * math.exp(-3*U_r) # vanish at high speed where quadratic
        D[1][1] = D[1][1] * math.exp(-3*U_r) # drag and lift forces dominates

        tau_liftdrag = self.forceLiftDrag(self.diam,self.S,self.CD_0,alpha,U_r)
        tau_crossflow = self.crossFlowDrag(self.L,self.diam,self.diam,nu_r)

        # Restoring forces and moments
        g = self.gvect(self.W,self.B,eta[4],eta[3],self.r_bg,self.r_bb)
        
        # Horizontal- and vertical-plane relative speed
        U_rh = math.sqrt( nu_r[0]**2 + nu_r[1]**2 )
        U_rv = math.sqrt( nu_r[0]**2 + nu_r[2]**2 ) 

        # Rudder and stern-plane drag
        X_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r**2
        X_s = -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_s**2

        # Rudder sway force 
        Y_r = -0.5 * self.rho * U_rh**2 * self.A_r * self.CL_delta_r * delta_r

        # Stern-plane heave force
        Z_s = -0.5 * self.rho * U_rv**2 * self.A_s * self.CL_delta_s * delta_s

        # Generalized force vector
        tau = np.array([
            (1-t_prop) * X_prop + X_r + X_s, 
            Y_r, 
            Z_s,
            K_prop / 10,   # scaled down by a factor of 10 to match exp. results
            -1 * self.x_s * Z_s,
            self.x_r * Y_r
            ], float)
    
        # AUV dynamics
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C+D,nu_r)  - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum)
            
        # Actuator dynamics
        delta_r_dot = (delta_r_c - delta_r) / self.T_delta
        delta_s_dot = (delta_s_c - delta_s) / self.T_delta
        n_dot = (n_c - n) / self.T_n

        # Forward Euler integration [k+1]
        nu += sampleTime * nu_dot
        delta_r += sampleTime * delta_r_dot
        delta_s += sampleTime * delta_s_dot
        n += sampleTime * n_dot
        
        u_actual = np.array([ delta_r, delta_s, n ], float)

        return nu, u_actual, nu_dot    
    
    def depthHeadingAutopilot(self, eta, nu, sampleTime):
        """
        [delta_r, delta_s, n] = depthHeadingAutopilot(eta,nu,sampleTime) 
        simultaneously control the heading and depth of the AUV using control
        laws of PID type. Propeller rpm is given as a step command.
        
        Returns:
            
            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]
            
        """
        self.current_position = [eta[0], eta[1], eta[2]]
        z = eta[2]                  # heave position (depth)
        theta = eta[4]              # pitch angle
        psi = eta[5]                # yaw angle
        w = nu[2]                   # heave velocity
        q = nu[4]                   # pitch rate
        r = nu[5]                   # yaw rate
        z_ref = self.ref_z          # heave position (depth) setpoint
        psi_ref = self.ref_psi * self.D2R   # yaw angle setpoint
        
        #######################################################################
        # Propeller command
        #######################################################################
        n = self.ref_n 
        
        #######################################################################            
        # Depth autopilot (succesive loop closure)
        #######################################################################
        # LP filtered desired depth command
        self.z_d  = math.exp( -sampleTime * self.wn_d_z ) * self.z_d \
            + ( 1 - math.exp( -sampleTime * self.wn_d_z) ) * z_ref  

        self.theta_d = self.depth_pi_controller(sampleTime)
        delta_s = self.theta_pi_controller(sampleTime)

        #######################################################################
        # Heading autopilot 
        #######################################################################
        self.psi_d = self.heading_control(nu)
        delta_r = self.yaw_pi_controller(sampleTime)

        if abs(delta_r) >= self.deltaMax_r:
            delta_r = np.sign(delta_r) * self.deltaMax_r
            
        if abs(delta_s) >= self.deltaMax_s:
            delta_s = np.sign(delta_s) * self.deltaMax_s          
            
        if abs(n) >= self.nMax:
            n = np.sign(n) * self.nMax   

        u_control = np.array([ delta_r, -delta_s, n], float)

        return u_control

    
    def trim(self, X0=None):
        V = self.ref_n / 1200
        V_knots = V / 0.514444 # m/s
        # Decision variables
        # Trim input vector indices
        # x_ind = [alpha beta phi theta n_prop u_prop delta_s delta_r Qm]
        x_ind = [17, 18, 9, 10, 12, 14, 13, 14, 16]
        # Trim output vector for generating Jacobian
        # xd_ind = [u v w p q r Z phi theta psi n_prop u_prop V alpha beta]
        xd_ind = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        if X0 is None:
            # Initial trim estimate
            X0 = [0, 0, 0, 0, self.ref_n, 0.9 * V, 0, 0, 39 * V]  # Analytical - prop only

        dx = 0.001  # perturbation size
        x_t = np.array(X0)  # initialise trim state vector
        error = 100  # initialise error
        tol = 1e-10  # error tolerance
        counter = 1  # initialise while loop counter

        # Main Loop -----------------------------------------------------------
        while error > tol:
            # Generate Jacobian -----------------------------------------------------
            J = np.zeros((len(xd_ind), len(x_ind)))
            for i in range(len(x_t)):
                x_tp = x_t.copy()  # reset trim vector
                x_tp[i] = x_t[i] + dx  # perturb i_th element in trim vector
                fx_tp = self.f(x_tp, x_ind, xd_ind, V)  # Calculate perturbed x_dot
                fx_t = self.f(x_t, x_ind, xd_ind, V)  # Calculate unperturbed x_dot
                J[:, i] = np.divide(fx_tp - fx_t,dx)  # calculate i_th column of Jacobian

            #------------------------------------------------------------------------
            x_t_old = x_t.copy()  # save old trim state
            solution = np.linalg.lstsq(J, fx_t, rcond=None)
            x_t = x_t - solution[0]  # calculate new trim state

            error = np.sum(np.abs(x_t - x_t_old))  # calculate error between new and old trim states
            counter += 1  # advance counter

            if counter > 1000:  # Break if calculated more than 100 iterations
                break

        trim_final = {}

        # calculate (u,v,w)
        trim_final['u_t'] = V_knots * np.cos(x_t[0]) * np.cos(x_t[1])
        trim_final['v_t'] = V_knots * np.sin(x_t[1])
        trim_final['w_t'] = V_knots * np.sin(x_t[0]) * np.cos(x_t[1])
        trim_final['alpha_t'] = x_t[0]
        trim_final['beta_t'] = x_t[1]
        trim_final['phi_t'] = x_t[2]
        trim_final['theta_t'] = x_t[3]
        trim_final['np_t'] = x_t[4]
        trim_final['up_t'] = x_t[5]
        trim_final['Q_t'] = x_t[8]
        trim_final['ds_t'] = x_t[6]
        trim_final['dr_t'] = x_t[7]

        # print(trim_final)
        # Display output
        iterations = counter - 1
        # print(f'iterations = {iterations}')
        # print(f'error = {error}')
        return trim_final, iterations, error

    def f(self, x, x_ind, xd_ind, V):
        # Function to manipulate the variables passed to and from the equation of motion function - eom
        X = np.zeros(19)  # initialise vehicle state vector
        X[x_ind] = x  # insert trim states into appropriate elements of vehicle state vector
        X_dot = self.eom(X, V)  # calculate vehicle state rate vector
        x_dot = X_dot[xd_ind]  # extract appropriate elements vehicle state rates vector for Jacobian
        return x_dot

    def eom(self, x, V):
        alpha = x[17]
        beta = x[18]
        u = V*np.cos(alpha)*np.cos(beta)
        v = V*np.sin(beta)
        w = V*np.sin(alpha)*np.cos(beta)
        p = x[3]
        q = x[4]
        r = x[5]

        nu = [u, v, w, p, q, r]

        nprop = self.ref_n 
        delta_s = x[13]
        delta_r = x[14]

        u_control = [delta_r, delta_s, nprop]
        u_actual = [delta_r, delta_s, nprop]

        xpos = x[6]
        ypos = x[7]
        zpos = x[8]
        phi = x[9]
        theta = x[10]
        psi = self.ref_psi * self.D2R

        eta = [xpos, ypos, zpos, phi, theta, psi]

        _, _, nu_dot = self.dynamics(eta, nu, u_actual, u_control, 0)
        
        x_dot = np.zeros(17) 

        x_dot[0:6] = nu_dot

        c1 = math.cos(phi)
        c2 = math.cos(theta)
        c3 = math.cos(psi)
        s1 = math.sin(phi)
        s2 = math.sin(theta)
        s3 = math.sin(psi)
        t2 = math.tan(theta)

        x_dot[6] = c3 * c2 * u + (c3 * s2 * s1 - s3 * c1) * v + (s3 * s1 + c3 * c1 * s2) * w
        x_dot[7] = s3 * c2 * u + (c1 * c3 + s1 * s2 * s3) * v + (c1 * s2 * s3-c3 * s1) * w
        x_dot[8] = -s2 * u + c2 * s1 * v + c1 * c2 * w
        x_dot[9] = p + s1 * t2 * q + c1 * t2 * r
        x_dot[10] = c1 * q - s1 * r
        x_dot[11] = s1 / c2 * q + c1 / c2 * r

        a = self.L / 2
        b = self.diam / 2                  
        m = 4/3 * math.pi * self.rho * a * b**2
        Ix = (2/5) * m * b**2

        x_dot[12] = x_dot[0]/m
        x_dot[13] = x_dot[3]/Ix * 2 * math.pi

        x_dot[14] = x_dot[0]*np.cos(alpha)*np.cos(beta) + x_dot[1]*np.sin(beta) + x_dot[2]*np.sin(alpha)*np.cos(beta)
        x_dot[15] = (x_dot[2]*np.cos(alpha) - x_dot[0]*np.sin(alpha))/(V*np.cos(beta))
        x_dot[16] = (1/V)*(-x_dot[0]*np.cos(alpha)*np.sin(beta) + x_dot[1]*np.cos(beta)-x_dot[2]*np.sin(alpha)*np.sin(beta))

        return x_dot

    def heading_control(self, nu):
        """Compute control commands based on current and target positions."""
        # Calculate desired heading
        desired_course = np.arctan2(self.target_position[1] - self.previous_target_position[1], 
                                    self.target_position[0] - self.previous_target_position[0])
        ye = (self.previous_target_position[0] - self.current_position[0]) * np.sin(desired_course) - (self.previous_target_position[1] - self.current_position[1]) * np.cos(desired_course)
        desired_heading = desired_course - np.arctan2(ye, 10) - np.arctan2(nu[1], nu[0])
        return desired_heading

    def yaw_pi_controller(self, sampleTime):
        error = (self.psi_d - self.eta[5] + np.pi) % (np.pi*2) - np.pi
        p_err = self.yaw_kp * error  
        self.e_psi_int += error * sampleTime
        i_err = self.yaw_ki * self.e_psi_int      
        output = p_err + i_err
        self.yaw_previous_error = error

        return output

    def depth_pi_controller(self, sampleTime):
        error = (self.eta[2]-self.z_d)
        p_err = self.z_kp * error  
        self.e_z_int += error * sampleTime
        i_err = self.z_ki * self.e_z_int      
        output = p_err + i_err
        self.z_previous_error = error

        return output

    def theta_pi_controller(self, sampleTime):
        error = (self.theta_d - self.eta[4] + np.pi) % (np.pi*2) - np.pi
        p_err = self.theta_kp * error  
        self.e_theta_int += error * sampleTime
        i_err = self.theta_ki * self.e_theta_int      
        output = p_err + i_err
        self.theta_previous_error = error

        return output

    def attitudeEuler(self, eta, nu, sampleTime):
   
        p_dot   = np.matmul(self.Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
        v_dot   = np.matmul(self.Tzyx(eta[3], eta[4]), nu[3:6] )

        p_dot = p_dot + np.random.normal(0.0, 0.25, 3)
        v_dot = v_dot + np.random.normal(0.0, 0.00025, 3)

        # Forward Euler integration
        eta[0:3] = eta[0:3] + sampleTime * p_dot
        eta[3:6] = eta[3:6] + sampleTime * v_dot

        return eta

    def Rzyx(self,phi,theta,psi):
        """
        R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
        using the zyx convention
        """
        
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        cth  = math.cos(theta)
        sth  = math.sin(theta)
        cpsi = math.cos(psi)
        spsi = math.sin(psi)
        
        R = np.array([
            [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
            [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
            [ -sth,      cth*sphi,                 cth*cphi ] ])

        return R

    def Tzyx(self,phi,theta):
        """
        T = Tzyx(phi,theta) computes the Euler angle attitude
        transformation matrix T using the zyx convention
        """
        
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        cth  = math.cos(theta)
        sth  = math.sin(theta)    

        try: 
            T = np.array([
                [ 1,  sphi*sth/cth,  cphi*sth/cth ],
                [ 0,  cphi,          -sphi],
                [ 0,  sphi/cth,      cphi/cth] ])
            
        except ZeroDivisionError:  
            print ("Tzyx is singular for theta = +-90 degrees." )
            
        return T
    
    def Hmtrx(self,r):
        """
        H = Hmtrx(r) computes the 6x6 system transformation matrix
        H = [eye(3)     S'
            zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)

        If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
        CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
        force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG 
        """
    
        H = np.identity(6,float)
        H[0:3, 3:6] = self.Smtrx(r).T

        return H

    def Smtrx(self,a):
        """
        S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
        The cross product satisfies: a x b = S(a)b. 
        """
    
        S = np.array([ 
            [ 0, -a[2], a[1] ],
            [ a[2],   0,     -a[0] ],
            [-a[1],   a[0],   0 ]  ])

        return S

    def Hoerner(self,B,T):
        """
        CY_2D = Hoerner(B,T)
        Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam 
        B and draft T.The data is digitized and interpolation is used to compute 
        other data point than those in the table
        """
        
        # DATA = [B/2T  C_D]
        DATA1 = np.array([
            0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
            0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031 
            ])
        DATA2 = np.array([
            1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
            0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593 
            ])

        CY_2D = np.interp( B / (2 * T), DATA1, DATA2 )
            
        return CY_2D

    def crossFlowDrag(self,L,B,T,nu_r):
        """
        tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
        integrals for a marine craft using strip theory. 

        M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
        """

        rho = 1026               # density of water
        n = 20                   # number of strips

        dx = L/20             
        Cd_2D = self.Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve

        Yh = 0
        Nh = 0
        xL = -L/2
        
        for i in range(0,n+1):
            v_r = nu_r[1]             # relative sway velocity
            r = nu_r[5]               # yaw rate
            Ucf = abs(v_r + xL * r) * (v_r + xL * r)
            Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
            Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
            xL += dx
            
        tau_crossflow = np.array([0, Yh, 0, 0, 0, Nh],float)

        return tau_crossflow

    def forceLiftDrag(self,b,S,CD_0,alpha,U_r):
        """
        tau_liftdrag = forceLiftDrag(b,S,CD_0,alpha,Ur) computes the hydrodynamic
        lift and drag forces of a submerged "wing profile" for varying angle of
        attack (Beard and McLain 2012). Application:
        
        M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_liftdrag
        
        Inputs:
            b:     wing span (m)
            S:     wing area (m^2)
            CD_0:  parasitic drag (alpha = 0), typically 0.1-0.2 for a streamlined body
            alpha: angle of attack, scalar or vector (rad)
            U_r:   relative speed (m/s)

        Returns:
            tau_liftdrag:  6x1 generalized force vector
        """

        # constants
        rho = 1026

        def coeffLiftDrag(b,S,CD_0,alpha,sigma):
            
            """
            [CL,CD] = coeffLiftDrag(b,S,CD_0,alpha,sigma) computes the hydrodynamic 
            lift CL(alpha) and drag CD(alpha) coefficients as a function of alpha
            (angle of attack) of a submerged "wing profile" (Beard and McLain 2012)

            CD(alpha) = CD_p + (CL_0 + CL_alpha * alpha)^2 / (pi * e * AR)
            CL(alpha) = CL_0 + CL_alpha * alpha
    
            where CD_p is the parasitic drag (profile drag of wing, friction and
            pressure drag of control surfaces, hull, etc.), CL_0 is the zero angle 
            of attack lift coefficient, AR = b^2/S is the aspect ratio and e is the  
            Oswald efficiency number. For lift it is assumed that
    
            CL_0 = 0
            CL_alpha = pi * AR / ( 1 + sqrt(1 + (AR/2)^2) );
    
            implying that for alpha = 0, CD(0) = CD_0 = CD_p and CL(0) = 0. For
            high angles of attack the linear lift model can be blended with a
            nonlinear model to describe stall
    
            CL(alpha) = (1-sigma) * CL_alpha * alpha + ...
                sigma * 2 * sign(alpha) * sin(alpha)^2 * cos(alpha) 

            where 0 <= sigma <= 1 is a blending parameter. 
            
            Inputs:
                b:       wing span (m)
                S:       wing area (m^2)
                CD_0:    parasitic drag (alpha = 0), typically 0.1-0.2 for a 
                        streamlined body
                alpha:   angle of attack, scalar or vector (rad)
                sigma:   blending parameter between 0 and 1, use sigma = 0 f
                        or linear lift 
                display: use 1 to plot CD and CL (optionally)
            
            Returns:
                CL: lift coefficient as a function of alpha   
                CD: drag coefficient as a function of alpha   

            Example:
                Cylinder-shaped AUV with length L = 1.8, diameter D = 0.2 and 
                CD_0 = 0.3
                
                alpha = 0.1 * pi/180
                [CL,CD] = coeffLiftDrag(0.2, 1.8*0.2, 0.3, alpha, 0.2)
            """
            
            e = 0.7             # Oswald efficiency number
            AR = b**2 / S       # wing aspect ratio

            # linear lift
            CL_alpha = math.pi * AR / ( 1 + math.sqrt(1 + (AR/2)**2) )
            CL = CL_alpha * alpha

            # parasitic and induced drag
            CD = CD_0 + CL**2 / (math.pi * e * AR)
            
            # nonlinear lift (blending function)
            CL = (1-sigma) * CL + sigma * 2 * np.sign(alpha) \
                * math.sin(alpha)**2 * math.cos(alpha)

            return CL, CD

        
        [CL, CD] = coeffLiftDrag(b,S,CD_0,alpha,0) 
        
        F_drag = 1/2 * rho * U_r**2 * S * CD    # drag force
        F_lift = 1/2 * rho * U_r**2 * S * CL    # lift force

        # transform from FLOW axes to BODY axes using angle of attack
        tau_liftdrag = np.array([
            math.cos(alpha) * (-F_drag) - math.sin(alpha) * (-F_lift),
            0,
            math.sin(alpha) * (-F_drag) + math.cos(alpha) * (-F_lift),
            0,
            0,
            0 ])

        return tau_liftdrag

    def m2c(self,M, nu):
        """
        C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
        mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
        """

        M = 0.5 * (M + M.T)     # systematization of the inertia matrix

        if (len(nu) == 6):      #  6-DOF model
        
            M11 = M[0:3,0:3]
            M12 = M[0:3,3:6] 
            M21 = M12.T
            M22 = M[3:6,3:6] 
        
            nu1 = nu[0:3]
            nu2 = nu[3:6]
            dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
            dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

            #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
            #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
            C = np.zeros( (6,6) )    
            C[0:3,3:6] = -self.Smtrx(dt_dnu1)
            C[3:6,0:3] = -self.Smtrx(dt_dnu1)
            C[3:6,3:6] = -self.Smtrx(dt_dnu2)
                
        else:   # 3-DOF model (surge, sway and yaw)
            #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
            #      0             0             M(1,1)*nu(1)
            #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
            C = np.zeros( (3,3) ) 
            C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
            C[1,2] =  M[0,0] * nu[0] 
            C[2,0] = -C[0,2]       
            C[2,1] = -C[1,2]
            
        return C

    def gvect(self,W,B,theta,phi,r_bg,r_bb):
        """
        g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring 
        forces about an arbitrarily point CO for a submerged body. 
        
        Inputs:
            W, B: weight and buoyancy (kg)
            phi,theta: roll and pitch angles (rad)
            r_bg = [x_g y_g z_g]: location of the CG with respect to the CO (m)
            r_bb = [x_b y_b z_b]: location of the CB with respect to th CO (m)
            
        Returns:
            g: 6x1 vector of restoring forces about CO
        """

        sth  = math.sin(theta)
        cth  = math.cos(theta)
        sphi = math.sin(phi)
        cphi = math.cos(phi)

        g = np.array([
            (W-B) * sth,
            -(W-B) * cth * sphi,
            -(W-B) * cth * cphi,
            -(r_bg[1]*W-r_bb[1]*B) * cth * cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
            (r_bg[2]*W-r_bb[2]*B) * sth         + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
            -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth      
            ])
        
        return g

    def addnoise(self, nu, u_actual):

        nu = nu + np.random.normal(0.0, 0.0025, self.nu.shape[0])
        u_actual = u_actual + np.random.normal(0.0, 0.0025, self.u_actual.shape[0])
        
        self.noise_counter += 1

        return nu, u_actual

    def water_currents(self):
        self.V_c = self.V_c0 + np.random.normal(0.05, 0.0025)
        self.beta_c = np.random.normal(0.0, 0.25)
        self.alpha_c = np.random.normal(0.0, 0.00025)


def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
        
    return angle 
        
# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen
"""

legendSize = 5  # legend size
figSize1 = [25, 13]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 150  # figure dpi value


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54


# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def plotVehicleStates(simTime, simData, filename, figNo):

    # Time vector
    t = simTime

    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    phi = R2D(ssa(simData[:, 3]))
    theta = R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = R2D(simData[:, 9])
    q = R2D(simData[:, 10])
    r = R2D(simData[:, 11])
    target_position_x = simData[:, 18]
    target_position_x[0] = 0
    target_position_y = simData[:, 19]
    target_position_y[0] = 0
    target_position_z = simData[:, 20]
    psi_d = R2D(simData[:, 21])

    # Speed
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    beta_c  = R2D(ssa(np.arctan2(v,u)))   # crab angle, beta_c    
    alpha_c = R2D(ssa(np.arctan2(w,u)))   # flight path angle
    chi = R2D(ssa(simData[:, 5] + np.arctan2(v, u)))  # course angle, chi=psi+beta_c

    # Plots
    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    # plt.grid()

    plt.subplot(3, 3, 1)
    plt.plot(y, x)
    plt.plot(target_position_y, target_position_x)
    plt.legend(["North-East positions (m)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 2)
    plt.plot(t, z, t, target_position_z)
    plt.legend(["Depth (m)", "Target Depth (m)"], fontsize=legendSize)
    plt.grid()

    plt.title("Vehicle states", fontsize=12)

    plt.subplot(3, 3, 3)
    plt.plot(t, phi, t, theta)
    plt.legend(["Roll angle (deg)", "Pitch angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 4)
    plt.plot(t, U)
    plt.legend(["Speed (m/s)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 5)
    plt.plot(t, psi, t, psi_d, t, chi)
    plt.legend(["Yaw angle (deg)", "Yaw Desired", "Course angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 6)
    plt.plot(t, theta, t, alpha_c)
    plt.legend(["Pitch angle (deg)", "Flight path angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 7)
    plt.plot(t, u, t, v, t, w)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(
        ["Surge velocity (m/s)", "Sway velocity (m/s)", "Heave velocity (m/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 8)
    plt.plot(t, p, t, q, t, r)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(
        ["Roll rate (deg/s)", "Pitch rate (deg/s)", "Yaw rate (deg/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 9)
    plt.plot(t, beta_c)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(["Crab angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.savefig(filename)

# plotControls(simTime, simData) plots the vehicle control inputs versus time
# in figure no. figNo
def plotControls(simTime, simData, filename, figNo):

    DOF = 6

    # Time vector
    t = simTime

    plt.figure(
        figNo, figsize=(4,6), dpi=dpiValue
    )

    # Columns and rows needed to plot vehicle.dimU control inputs
    col = 1
    row = int(math.ceil(3 / col))

    controls = [
            "Tail rudder (deg)",
            "Stern plane (deg)",
            "Propeller revolution (rpm)"
            ]

    # Plot the vehicle.dimU active control inputs
    for i in range(0, 3):

        u_control = simData[:, 2 * DOF + i]  # control input, commands
        u_actual = simData[:, 2 * DOF + 3 + i]  # actual control input

        if controls[i].find("deg") != -1:  # convert angles to deg
            u_control = R2D(u_control)
            u_actual = R2D(u_actual)

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend(
            [controls[i] + ", command", controls[i] + ", actual"],
            fontsize=legendSize,
        )
        plt.xlabel("Time (s)", fontsize=12)
        plt.grid()
        plt.ticklabel_format(useOffset=False)

    plt.savefig(filename)


# plot3D(simData,numDataPoints,FPS,filename,figNo) plots the vehicles position (x, y, z) in 3D
# in figure no. figNo
def plot3D(simData,target_positions,numDataPoints,FPS,filename,figNo):
        
    # State vectors
    x = simData[:,0]
    y = simData[:,1]
    z = simData[:,2]
    
    # down-sampling the xyz data points
    N = y[::len(x) // numDataPoints]
    E = x[::len(x) // numDataPoints]
    D = z[::len(x) // numDataPoints]
    
    # Animation function
    def anim_function(num, dataSet, line):
        
        line.set_data(dataSet[0:2, :num])    
        line.set_3d_properties(dataSet[2, :num])    
        ax.view_init(elev=10.0, azim=-120.0)
        
        return line
    
    dataSet = np.array([N, E, -D])      # Down is negative z
    
    # Attaching 3D axis to the figure
    fig = plt.figure(figNo,figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
               dpi=dpiValue)
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax) 
    
    # Line/trajectory plot
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b')[0] 

    # Setting the axes properties
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlim3d([-100, 20])                   # default depth = -100 m
    
    if np.amax(z) > 100.0:
        ax.set_zlim3d([-np.amax(z), 20])
        
    ax.set_zlabel('-Z / Down')

    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx

    plot_targets = np.array(target_positions)
    ax.scatter(plot_targets[:,1],plot_targets[:,0],-1*plot_targets[:,2])
    ax.plot_surface(xx, yy, zz, alpha=0.3)
                    
    # Title of plot
    ax.set_title('North-East-Down')
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=numDataPoints, 
                         fargs=(dataSet,line),
                         interval=200, 
                         blit=False,
                         repeat=True)
    
    # Save the 3D animation as a gif file
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))  

def plot_controls_2D(simData, FPS, filename, figNo):

    rudder_command = R2D(simData[:, 12])
    rudder_actual = R2D(simData[:, 15])
    stern_command = R2D(simData[:, 13])
    stern_actual = R2D(simData[:, 16])
    
    # Animation function
    def anim_function(i): 
        ax.clear()
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

        # Setting the axes properties
        ax.set_xlabel('Tail rudder (deg)')
        ax.set_ylabel('Stern plane (deg)')
                        
        # Title of plot
        ax.set_title('AUV Controls (10x speed)')    
        stern = plt.Rectangle((-20, 0), 40, stern_actual[i], fc='red')
        ax.add_patch(stern)
        rudder = plt.Rectangle((0, -20), rudder_actual[i], 40, fc='red')
        ax.add_patch(rudder)
        ax.plot([-30, 30], [0, 0], linestyle='dotted', color='black')
        ax.plot([0, 0], [-30, 30], linestyle='dotted', color='black')

    fig, ax = plt.subplots(num=figNo)
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=len(rudder_actual), 
                         interval=20)
    # Save the 3D animation as a gif file
    ani.save(filename, dpi=80, writer='pillow') 

def plot_attitude_2D(simData, FPS, filename, figNo):

    theta = R2D(simData[:, 4])
    psi = R2D(simData[:, 5])

    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]

    beta = R2D(ssa(np.arctan2(v,u))) 
    alpha = R2D(ssa(np.arctan2(w,u)))

    # Animation function
    def anim_function(i): 
        ax.clear()
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)

        # Setting the axes properties
        ax.set_xlabel('Ref Yaw / Crab (deg)')
        ax.set_ylabel('Pitch / Flight Path (deg)')

        # Title of plot
        ax.set_title('AUV Attitudes (10x speed))')    
        ax.scatter(0, theta[i], marker='o', facecolor='none', edgecolor='blue', label='Pitch / Ref Yaw')
        ax.scatter(beta[i], alpha[i], marker='o', facecolor='none', edgecolor='red', label='Crab / Flight')

        ax.plot([-40, 40], [0, 0], linestyle='dotted', color='black')
        ax.plot([0, 0], [-40, 40], linestyle='dotted', color='black')
        ax.legend()

    fig, ax = plt.subplots(num=figNo)
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=len(theta), 
                         interval=20)
    # Save the 3D animation as a gif file
    ani.save(filename, dpi=80, writer='pillow') 

# main()