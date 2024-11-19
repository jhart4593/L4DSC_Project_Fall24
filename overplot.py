from remus100 import R2D, ssa, cm2inch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from config import config
from eval_config import eval_config

legendSize = 5  # legend size
figSize1 = [25, 13]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 150  # figure dpi value

def overplotVehicleTimeHistory(ppoTime, ppoData, refTime, refData, sacTime, sacData, filename, figNo):

    # Time vector
    t_PPO = ppoTime

    # State vectors
    x_PPO = ppoData[:, 0]
    y_PPO = ppoData[:, 1]
    z_PPO = ppoData[:, 2]
    phi_PPO = R2D(ssa(ppoData[:, 3]))
    theta_PPO = R2D(ssa(ppoData[:, 4]))
    psi_PPO = R2D(ssa(ppoData[:, 5]))
    u_PPO = ppoData[:, 6]
    v_PPO = ppoData[:, 7]
    w_PPO = ppoData[:, 8]
    p_PPO = R2D(ppoData[:, 9])
    q_PPO = R2D(ppoData[:, 10])
    r_PPO = R2D(ppoData[:, 11])
    target_position_x_PPO = ppoData[:, 18]
    target_position_x_PPO[0] = 0
    target_position_y_PPO = ppoData[:, 19]
    target_position_y_PPO[0] = 0
    target_position_z_PPO = ppoData[:, 20]
    psi_d_PPO = R2D(ppoData[:, 21])
    theta_d_PPO = R2D(ppoData[:, 22])

    # Speed
    U_PPO = np.sqrt(np.multiply(u_PPO, u_PPO) + np.multiply(v_PPO, v_PPO) + np.multiply(w_PPO, w_PPO))

    beta_c_PPO  = R2D(ssa(np.arctan2(v_PPO,u_PPO)))   # crab angle, beta_c    
    alpha_c_PPO = R2D(ssa(np.arctan2(w_PPO,u_PPO)))   # flight path angle
    chi_PPO = R2D(ssa(ppoData[:, 5] + np.arctan2(v_PPO, u_PPO)))  # course angle, chi=psi+beta_c

    # Controls
    rud_PPO = R2D(ppoData[:, 12])
    rud_cmd_PPO = R2D(ppoData[:, 15])
    stern_PPO = R2D(ppoData[:, 13])
    stern_cmd_PPO = R2D(ppoData[:, 16])
    prop_PPO = ppoData[:, 14]
    prop_cmd_PPO = ppoData[:, 17]

    # Time vector
    t_ref = refTime

    # State vectors
    x_ref = refData[:, 0]
    y_ref = refData[:, 1]
    z_ref = refData[:, 2]
    phi_ref = R2D(ssa(refData[:, 3]))
    theta_ref = R2D(ssa(refData[:, 4]))
    psi_ref = R2D(ssa(refData[:, 5]))
    u_ref = refData[:, 6]
    v_ref = refData[:, 7]
    w_ref = refData[:, 8]
    p_ref = R2D(refData[:, 9])
    q_ref = R2D(refData[:, 10])
    r_ref = R2D(refData[:, 11])
    target_position_x_ref = refData[:, 18]
    target_position_x_ref[0] = 0
    target_position_y_ref = refData[:, 19]
    target_position_y_ref[0] = 0
    target_position_z_ref = refData[:, 20]
    psi_d_ref = R2D(refData[:, 21])
    theta_d_ref = R2D(refData[:, 22])

    # Speed
    U_ref = np.sqrt(np.multiply(u_ref, u_ref) + np.multiply(v_ref, v_ref) + np.multiply(w_ref, w_ref))

    beta_c_ref  = R2D(ssa(np.arctan2(v_ref,u_ref)))   # crab angle, beta_c    
    alpha_c_ref = R2D(ssa(np.arctan2(w_ref,u_ref)))   # flight path angle
    chi_ref = R2D(ssa(refData[:, 5] + np.arctan2(v_ref, u_ref)))  # course angle, chi=psi+beta_c

    # Controls
    rud_ref = R2D(refData[:, 12])
    rud_cmd_ref = R2D(refData[:, 15])
    stern_ref = R2D(refData[:, 13])
    stern_cmd_ref = R2D(refData[:, 16])
    prop_ref = refData[:, 14]
    prop_cmd_ref = refData[:, 17]

    if sacTime is not None:
        # Time vector
        t_SAC = sacTime

        # State vectors
        x_SAC = sacData[:, 0]
        y_SAC = sacData[:, 1]
        z_SAC = sacData[:, 2]
        phi_SAC = R2D(ssa(sacData[:, 3]))
        theta_SAC = R2D(ssa(sacData[:, 4]))
        psi_SAC = R2D(ssa(sacData[:, 5]))
        u_SAC = sacData[:, 6]
        v_SAC = sacData[:, 7]
        w_SAC = sacData[:, 8]
        p_SAC = R2D(sacData[:, 9])
        q_SAC = R2D(sacData[:, 10])
        r_SAC = R2D(sacData[:, 11])
        target_position_x_SAC = sacData[:, 18]
        target_position_x_SAC[0] = 0
        target_position_y_SAC = sacData[:, 19]
        target_position_y_SAC[0] = 0
        target_position_z_SAC = sacData[:, 20]
        psi_d_SAC = R2D(sacData[:, 21])
        theta_d_SAC = R2D(sacData[:, 22])

        # Speed
        U_SAC = np.sqrt(np.multiply(u_SAC, u_SAC) + np.multiply(v_SAC, v_SAC) + np.multiply(w_SAC, w_SAC))

        beta_c_SAC  = R2D(ssa(np.arctan2(v_SAC,u_SAC)))   # crab angle, beta_c    
        alpha_c_SAC = R2D(ssa(np.arctan2(w_SAC,u_SAC)))   # flight path angle
        chi_SAC = R2D(ssa(sacData[:, 5] + np.arctan2(v_SAC, u_SAC)))  # course angle, chi=psi+beta_c

        # Controls
        rud_SAC = R2D(sacData[:, 12])
        rud_cmd_SAC = R2D(sacData[:, 15])
        stern_SAC = R2D(sacData[:, 13])
        stern_cmd_SAC = R2D(sacData[:, 16])
        prop_SAC = sacData[:, 14]
        prop_cmd_SAC = sacData[:, 17]

    # Plots
    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    # plt.grid()

    plt.subplot(3, 3, 1)
    plt.plot(target_position_y_PPO, target_position_x_PPO, color='seagreen')
    plt.plot(y_PPO, x_PPO, color='limegreen')
    plt.plot(target_position_y_ref, target_position_x_ref, color='darkblue')
    plt.plot(y_ref, x_ref, color='cornflowerblue')
    legend_list = ["PPO Target", "PPO", "Manual Target", "Manual"]
    if sacTime is not None:
        plt.plot(target_position_y_SAC, target_position_x_SAC, color='maroon')
        plt.plot(y_SAC, x_SAC, color='lightcoral') 
        legend_list.extend(["SAC Target", "SAC"])      
    plt.ylabel("North-East positions (m)")
    plt.legend(legend_list, fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 4)
    plt.plot(t_PPO, target_position_z_PPO, color='seagreen')
    plt.plot(t_PPO, z_PPO, color='limegreen')
    plt.plot(t_ref, target_position_z_ref, color='darkblue')
    plt.plot(t_ref, z_ref, color='cornflowerblue')
    legend_list = ["PPO Target", "PPO", "Manual Target", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, target_position_z_SAC, color='maroon')
        plt.plot(t_SAC, z_SAC, color='lightcoral')
        legend_list.extend(["SAC Target", "SAC"])   
    plt.ylabel("Depth (m)")
    plt.legend(legend_list, fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 7)
    plt.plot(t_PPO, U_PPO, color='limegreen')
    plt.plot(t_ref, U_ref, color='cornflowerblue')
    legend_list = ["PPO", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, U_SAC, color='lightcoral')
        legend_list.extend(["SAC"])   
    plt.ylabel("Speed (m/s)")
    plt.legend(legend_list, fontsize=legendSize) 
    plt.grid()

    plt.subplot(3, 3, 2)
    plt.plot(t_PPO, phi_PPO, color='limegreen')
    plt.plot(t_ref, phi_ref, color='cornflowerblue')
    legend_list = ["PPO", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, phi_SAC, color='lightcoral')
        legend_list.extend(["SAC"])  
    plt.ylabel("Roll (deg)")
    plt.legend(legend_list, fontsize=legendSize) 
    plt.grid()
    
    plt.subplot(3, 3, 5)
    plt.plot(t_PPO, theta_d_PPO, color='seagreen')
    plt.plot(t_PPO, theta_PPO, color='limegreen')
    plt.plot(t_ref, theta_d_ref, color='darkblue')
    plt.plot(t_ref, theta_ref, color='cornflowerblue')
    legend_list = ["PPO Desired", "PPO", "Manual Desired", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, theta_d_SAC, color='maroon')
        plt.plot(t_SAC, theta_SAC, color='lightcoral')
        legend_list.extend(["SAC Desired", "SAC"])
    plt.ylabel("Pitch (deg)")
    plt.legend(legend_list, fontsize=legendSize) 
    plt.grid()

    plt.subplot(3, 3, 8)
    plt.plot(t_PPO, psi_d_PPO, color='seagreen')
    plt.plot(t_PPO, psi_PPO, color='limegreen') 
    plt.plot(t_ref, psi_d_ref, color='darkblue')
    plt.plot(t_ref, psi_ref, color='cornflowerblue')
    legend_list = ["PPO Desired", "PPO", "Manual Desired", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, psi_d_SAC, color='maroon')
        plt.plot(t_SAC, psi_SAC, color='lightcoral')
        legend_list.extend(["SAC Desired", "SAC"])   
    plt.ylabel("Yaw (deg)")
    plt.legend(legend_list, fontsize=legendSize) 
    plt.grid()

    plt.subplot(3, 3, 3)
    plt.plot(t_PPO, rud_cmd_PPO, color='seagreen')
    plt.plot(t_PPO, rud_PPO, color='limegreen')
    plt.plot(t_ref, rud_cmd_ref, color='darkblue')
    plt.plot(t_ref, rud_ref, color='cornflowerblue')
    legend_list = ["PPO Command", "PPO", "Manual Command", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, rud_cmd_SAC, color='maroon')
        plt.plot(t_SAC, rud_SAC, color='lightcoral')
        legend_list.extend(["SAC Command", "SAC"])  
    plt.ylabel("Rudder (deg)")
    plt.legend(legend_list, fontsize=legendSize) 
    plt.grid()
    
    plt.subplot(3, 3, 6)
    plt.plot(t_PPO, stern_cmd_PPO, color='seagreen')
    plt.plot(t_PPO, stern_PPO, color='limegreen')
    plt.plot(t_ref, stern_cmd_ref, color='darkblue')
    plt.plot(t_ref, stern_ref, color='cornflowerblue')
    legend_list = ["PPO Command", "PPO", "Manual Command", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, stern_cmd_SAC, color='maroon')
        plt.plot(t_SAC, stern_SAC, color='lightcoral')
        legend_list.extend(["SAC Command", "SAC"])  
    plt.ylabel("Stern plane (deg)")
    plt.legend(legend_list, fontsize=legendSize) 
    plt.grid()

    plt.subplot(3, 3, 9)
    plt.plot(t_PPO, prop_cmd_PPO, color='seagreen')
    plt.plot(t_PPO, prop_PPO, color='limegreen')
    plt.plot(t_ref, prop_cmd_ref, color='darkblue')
    plt.plot(t_ref, prop_ref, color='cornflowerblue') 
    legend_list = ["PPO Command", "PPO", "Manual Command", "Manual"]
    if sacTime is not None:
        plt.plot(t_SAC, prop_cmd_SAC, color='maroon')
        plt.plot(t_SAC, prop_SAC, color='lightcoral')
        legend_list.extend(["SAC Command", "SAC"])  
    plt.ylabel("Propeller (rpm)")
    plt.legend(legend_list, fontsize=legendSize) 
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.grid()

    plt.tight_layout()

    plt.savefig(filename)

def overplot3D(ppoData, refData, sacData, target_positions, numDataPoints, FPS, filename, figNo):
        
    # State vectors
    x_PPO = ppoData[:, 0]
    y_PPO = ppoData[:, 1]
    z_PPO = ppoData[:, 2]
    x_ref = refData[:, 0]
    y_ref = refData[:, 1]
    z_ref = refData[:, 2]
    if sacTime is not None:
        x_SAC = sacData[:, 0]
        y_SAC = sacData[:, 1]
        z_SAC = sacData[:, 2]

    # down-sampling the xyz data points
    N_PPO = y_PPO[::len(x_PPO) // numDataPoints]
    E_PPO = x_PPO[::len(x_PPO) // numDataPoints]
    D_PPO = z_PPO[::len(x_PPO) // numDataPoints]
    N_ref = y_ref[::len(x_ref) // numDataPoints]
    E_ref = x_ref[::len(x_ref) // numDataPoints]
    D_ref = z_ref[::len(x_ref) // numDataPoints]
    if sacTime is not None:
        N_SAC = y_SAC[::len(x_SAC) // numDataPoints]
        E_SAC = x_SAC[::len(x_SAC) // numDataPoints]
        D_SAC = z_SAC[::len(x_SAC) // numDataPoints]

    # Animation function
    def anim_function(num, dataSet_PPO, dataSet_ref, dataSet_SAC, line_PPO, line_ref, line_SAC):
        
        line_PPO.set_data(dataSet_PPO[0:2, :num])    
        line_PPO.set_3d_properties(dataSet_PPO[2, :num])    
        line_ref.set_data(dataSet_ref[0:2, :num])    
        line_ref.set_3d_properties(dataSet_ref[2, :num])
        if line_SAC is not None:
            line_SAC.set_data(dataSet_SAC[0:2, :num])    
            line_SAC.set_3d_properties(dataSet_SAC[2, :num])
        ax.view_init(elev=10.0, azim=-120.0)
        
        return
    
    dataSet_PPO = np.array([N_PPO, E_PPO, -D_PPO])      # Down is negative z
    dataSet_ref = np.array([N_ref, E_ref, -D_ref])      # Down is negative z
    if sacTime is not None:
        dataSet_SAC = np.array([N_SAC, E_SAC, -D_SAC])      # Down is negative z
    else:
        dataSet_SAC = None

    # Attaching 3D axis to the figure
    fig = plt.figure(figNo,figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
               dpi=dpiValue)
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax) 
    
    # Line/trajectory plot
    line_PPO = plt.plot(dataSet_PPO[0], dataSet_PPO[1], dataSet_PPO[2], lw=2, c='limegreen')[0] 
    line_ref = plt.plot(dataSet_ref[0], dataSet_ref[1], dataSet_ref[2], lw=2, c='cornflowerblue', linestyle='--')[0]
    if sacTime is not None:
        line_SAC = plt.plot(dataSet_SAC[0], dataSet_SAC[1], dataSet_SAC[2], lw=2, c='lightcoral')[0]
    else:
        line_SAC = None

    # Setting the axes properties
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlim3d([-75, 20]) # default depth = -100 m
    
    if np.amax(z_PPO) > 100.0:
        ax.set_zlim3d([-np.amax(z), 20])
        
    ax.set_zlabel('-Z / Down')

    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx

    plot_targets = np.array(target_positions)
    ax.scatter(plot_targets[:,1],plot_targets[:,0],-1*plot_targets[:,2],color = "black")
    ax.plot_surface(xx, yy, zz, alpha=0.3)
                    
    # Title of plot
    ax.set_title('North-East-Down')
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=numDataPoints, 
                         fargs=(dataSet_PPO,dataSet_ref,dataSet_SAC,line_PPO,line_ref,line_SAC),
                         interval=200, 
                         blit=False,
                         repeat=True)
    
    # Save the 3D animation as a gif file
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))  

def overplot_controls_2D(ppoData, refData, sacData, FPS, filename, figNo):

    rud_cmd_PPO = R2D(ppoData[:, 12])
    stern_cmd_PPO = R2D(ppoData[:, 13])
    rud_cmd_ref = R2D(refData[:, 12])
    stern_cmd_ref = R2D(refData[:, 13])
    if sacData is not None:
        rud_cmd_SAC = R2D(sacData[:, 12])
        stern_cmd_SAC = R2D(sacData[:, 13])

    # Animation function
    def anim_function(i): 
        ax[0].clear()
        ax[0].set_xlim(-30, 30)
        ax[0].set_ylim(-30, 30)

        # Setting the axes properties
        ax[0].set_xlabel('Tail rudder (deg)')
        ax[0].set_ylabel('Stern plane (deg)')
                        
        # Title of plot
        ax[0].set_title('Manual PID')    
        stern = plt.Rectangle((-20, 0), 40, stern_cmd_ref[i], fc='cornflowerblue')
        ax[0].add_patch(stern)
        rudder = plt.Rectangle((0, -20), rud_cmd_ref[i], 40, fc='cornflowerblue')
        ax[0].add_patch(rudder)
        ax[0].plot([-30, 30], [0, 0], linestyle='dotted', color='black')
        ax[0].plot([0, 0], [-30, 30], linestyle='dotted', color='black')
        ax[0].set_box_aspect(1)

        ax[1].clear()
        ax[1].set_xlim(-30, 30)
        ax[1].set_ylim(-30, 30)

        # Setting the axes properties
        ax[1].set_xlabel('Tail rudder (deg)')
        ax[1].set_ylabel('Stern plane (deg)')
                        
        # Title of plot
        ax[1].set_title('PPO')    
        stern = plt.Rectangle((-20, 0), 40, stern_cmd_PPO[i], fc='limegreen')
        ax[1].add_patch(stern)
        rudder = plt.Rectangle((0, -20), rud_cmd_PPO[i], 40, fc='limegreen')
        ax[1].add_patch(rudder)
        ax[1].plot([-30, 30], [0, 0], linestyle='dotted', color='black')
        ax[1].plot([0, 0], [-30, 30], linestyle='dotted', color='black')
        ax[1].set_box_aspect(1)

        if sacData is not None:
            ax[2].clear()
            ax[2].set_xlim(-30, 30)
            ax[2].set_ylim(-30, 30)

            # Setting the axes properties
            ax[2].set_xlabel('Tail rudder (deg)')
            ax[2].set_ylabel('Stern plane (deg)')
                            
            # Title of plot
            ax[2].set_title('SAC')    
            stern = plt.Rectangle((-20, 0), 40, stern_cmd_SAC[i], fc='lightcoral')
            ax[2].add_patch(stern)
            rudder = plt.Rectangle((0, -20), rud_cmd_SAC[i], 40, fc='lightcoral')
            ax[2].add_patch(rudder)
            ax[2].plot([-30, 30], [0, 0], linestyle='dotted', color='black')
            ax[2].plot([0, 0], [-30, 30], linestyle='dotted', color='black')
            ax[2].set_box_aspect(1)

    if sacData is not None:
        fig, ax = plt.subplots(1, 3, figsize=(8, 3.5), num=figNo, sharey=True)
        numpoints = min(len(rud_cmd_PPO),len(rud_cmd_ref),len(rud_cmd_SAC))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3.5), num=figNo, sharey=True)
        numpoints = min(len(rud_cmd_PPO),len(rud_cmd_ref))
    fig.suptitle('AUV Controls (10x speed)')    

    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=numpoints, 
                         interval=20)
    # Save the 3D animation as a gif file
    ani.save(filename, dpi=80, writer='pillow') 

def overplot_attitude_2D(ppoData, refData, sacData, FPS, filename, figNo):

    theta_PPO = R2D(ppoData[:, 4])
    psi_PPO = R2D(ppoData[:, 5])
    u_PPO = ppoData[:, 6]
    v_PPO = ppoData[:, 7]
    w_PPO = ppoData[:, 8]
    beta_PPO = R2D(ssa(np.arctan2(v_PPO,u_PPO))) 
    alpha_PPO = R2D(ssa(np.arctan2(w_PPO,u_PPO)))

    theta_ref = R2D(refData[:, 4])
    psi_ref = R2D(refData[:, 5])
    u_ref = refData[:, 6]
    v_ref = refData[:, 7]
    w_ref = refData[:, 8]
    beta_ref = R2D(ssa(np.arctan2(v_ref,u_ref))) 
    alpha_ref = R2D(ssa(np.arctan2(w_ref,u_ref)))

    if sacData is not None:
        theta_SAC = R2D(sacData[:, 4])
        psi_SAC = R2D(sacData[:, 5])
        u_SAC = sacData[:, 6]
        v_SAC = sacData[:, 7]
        w_SAC = sacData[:, 8]
        beta_SAC = R2D(ssa(np.arctan2(v_SAC,u_SAC))) 
        alpha_SAC = R2D(ssa(np.arctan2(w_SAC,u_SAC)))

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
        ax.scatter(0, theta_ref[i], marker='o', facecolor='darkblue', edgecolor='darkblue', label='Manual Pitch / Ref Yaw')
        ax.scatter(beta_ref[i], alpha_ref[i], marker='o', facecolor='cornflowerblue', edgecolor='cornflowerblue', label='Manual Crab / Flight')

        ax.scatter(0, theta_PPO[i], marker='o', facecolor='seagreen', edgecolor='seagreen', label='PPO Pitch / Ref Yaw')
        ax.scatter(beta_PPO[i], alpha_PPO[i], marker='o', facecolor='limegreen', edgecolor='limegreen', label='PPO Crab / Flight')
        
        if sacData is not None:
            ax.scatter(0, theta_PPO[i], marker='o', facecolor='Maroon', edgecolor='Maroon', label='SAC Pitch / Ref Yaw')
            ax.scatter(beta_PPO[i], alpha_PPO[i], marker='o', facecolor='lightcoral', edgecolor='lightcoral', label='SAC Crab / Flight')

        ax.plot([-40, 40], [0, 0], linestyle='dotted', color='black')
        ax.plot([0, 0], [-40, 40], linestyle='dotted', color='black')
        ax.legend()

    if sacData is not None:
        numpoints = min(len(theta_PPO),len(theta_ref),len(theta_SAC))
    else:
        numpoints = min(len(theta_PPO),len(theta_ref))
    fig, ax = plt.subplots(num=figNo)
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=numpoints, 
                         interval=20)
    # Save the 3D animation as a gif file
    ani.save(filename, dpi=80, writer='pillow') 

def violin_plot(ppoData, refData, sacData, FPS, filename, figNo):

    # Calculate errors from RL model vs. PID model
    yaw_err_ref = R2D(refData[:,21] - refData[:,5])
    yaw_err_PPO = R2D(ppoData[:,21] - ppoData[:,5])
    if sacData is not None:
        yaw_err_SAC = R2D(sacData[:,21] - sacData[:,5])

    depth_err_ref = refData[:,20] - refData[:,2]
    depth_err_PPO = ppoData[:,20] - ppoData[:,2]
    if sacData is not None:
        depth_err_SAC = sacData[:,20] - sacData[:,2]

    pitch_err_ref = R2D(refData[:,22] - refData[:,4])
    pitch_err_PPO = R2D(ppoData[:,22] - ppoData[:,4])
    if sacData is not None:
        pitch_err_SAC = R2D(sacData[:,22] - sacData[:,4])

    if sacData is not None:
        labels = ['Manual', 'PPO', 'SAC']
    else:
        labels = ['Manual', 'PPO']
    colors = ['cornflowerblue', 'limegreen', 'lightcoral']

    plt.figure(figNo, dpi=dpiValue)

    plt.subplot(1, 3, 1)
    plt.xlabel('Method')
    plt.ylabel('Yaw Angle Error')

    if sacData is not None:
        parts = plt.violinplot([yaw_err_ref,yaw_err_PPO,yaw_err_SAC],showextrema=False)
    else:
        parts = plt.violinplot([yaw_err_ref,yaw_err_PPO],showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
    plt.xlim(0.25, len(labels) + 0.75)

    plt.subplot(1, 3, 2)
    plt.xlabel('Method')
    plt.ylabel('Depth Error')

    if sacData is not None:
        parts = plt.violinplot([depth_err_ref,depth_err_PPO,depth_err_SAC],showextrema=False)
    else:
        parts = plt.violinplot([depth_err_ref,depth_err_PPO],showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
    plt.xlim(0.25, len(labels) + 0.75)

    plt.subplot(1, 3, 3)
    plt.xlabel('Method')
    plt.ylabel('Pitch Error')

    if sacData is not None:
        parts = plt.violinplot([pitch_err_ref,pitch_err_PPO,pitch_err_SAC],showextrema=False)
    else:
        parts = plt.violinplot([pitch_err_ref,pitch_err_PPO],showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
    plt.xlim(0.25, len(labels) + 0.75)

    plt.tight_layout()
    plt.savefig(filename)

if __name__=="__main__":
    ppoData = np.genfromtxt('PPO_AUV_eval.csv', delimiter=',')
    ppoTime = []
    for i in range(ppoData.shape[0]):
        t = i * config["sim_dt"]
        ppoTime.append(t)

    refImport = np.loadtxt('REMUS100_Reference.csv', delimiter=',') 
    refTime = refImport[:, 0]
    refData = refImport[:, 1:]

    # sacData = None
    # sacTime = None
    sacData = np.genfromtxt('SAC_AUV_eval.csv', delimiter=',')
    sacTime = []
    for i in range(sacData.shape[0]):
        t = i * config["sim_dt"]
        sacTime.append(t)

    target_positions = eval_config["path"]

    overplotVehicleTimeHistory(ppoTime, ppoData, refTime, refData, sacTime, sacData, 'AUV_eval_time_history_overplot.png', 2)
    overplot3D(ppoData, refData, sacData, target_positions, 100, 10, 'AUV_eval_3D_overplot.gif', 4)  
    overplot_controls_2D(ppoData, refData, sacData, 50, 'AUV_eval_controls_2D_overplot.gif', 5)
    overplot_attitude_2D(ppoData, refData, sacData, 50, 'AUV_eval_attitude_2D_overplot.gif', 6)
    violin_plot(ppoData, refData, sacData, 50, 'AUV_eval_violin_plot.png', 7)

