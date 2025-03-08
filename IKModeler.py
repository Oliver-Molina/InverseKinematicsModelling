import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import functools
from IKEngine import *
from CanisArmModel import *
from SSRTArm import *
from SSRTArm2025v1 import SSRTArmWrapper
import sys

frames = 1000
interval = 50

save_animation = False

# Initialize targets and orientations
engine = None
target_positions = np.array([None])        # x, y, z
target_orientations = np.array([None])   # thetax, thetay, thetaz
position_step = 0.01  # Increment/decrement step
orientation_step = math.pi/50
positions_enabled = False
orientations_enabled = False



# Key mapping
key_mappings = {
    "1": (True, 0, position_step),       # Increase X
    "2": (True, 1, position_step),       # Increase Y
    "3": (True, 2, position_step),       # Increase Z
    "4": (False, 0, orientation_step),  # Increase ThetaX
    "5": (False, 1, orientation_step),  # Increase ThetaY
    "6": (False, 2, orientation_step),  # Increase ThetaZ
    
    "!": (True, 0, -position_step),      # Decrease X (Shift+1)
    "@": (True, 1, -position_step),      # Decrease Y (Shift+2)
    "#": (True, 2, -position_step),      # Decrease Z (Shift+3)
    "$": (False, 0, -orientation_step), # Decrease ThetaX (Shift+4)
    "%": (False, 1, -orientation_step), # Decrease ThetaY (Shift+5)
    "^": (False, 2, -orientation_step)  # Decrease ThetaZ (Shift+6)
}

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def magnitude(x):
    return math.sqrt(sum(i**2 for i in x))

def get_colour(i):
    colours = ['b', 'g', 'r', 'c', 'm', 'y']
    return colours[i % len(colours)]

def key_handler(event):
    global target_positions, target_orientations, engine, positions_enabled, orientations_enabled
    
    if event.key in key_mappings:
        is_target, idx, delta = key_mappings[event.key]

        if is_target:
            for target in [t for t in target_positions if t is not None]:
                target[idx] += delta
        else:
            for orientation in [o for o in target_orientations if o is not None]:
                orientation[idx] += delta

        print(f"Target: {target_positions} Orientation: {target_orientations}")
            
    elif event.key == "7": # Toggle Target
        positions_enabled = not positions_enabled
        if positions_enabled:
            target_positions = np.array(engine.getEndEffectorTargetPositions())
        else:
            target_positions = np.full(len(target_positions), None, dtype=object)

        print("Target " + ("enabled" if positions_enabled else "disabled") + ".")


    elif event.key == "8":  # Toggle orientations between None and default
        orientations_enabled = not orientations_enabled
        if orientations_enabled:
            target_orientations = np.array(engine.getEndEffectorTargetOrientations())
        else:
            target_orientations = np.full(len(target_orientations), None, dtype=object)

        print("Orientation " + ("enabled" if orientations_enabled else "disabled") + ".")

def updatePlot(frame, animationEnabled):
    global engine, target_positions, target_orientations
    animatedTargets = target_positions
    animatedOrientations = target_orientations

    if animationEnabled:
        for i in range(animatedTargets.shape[0]):
            offset = math.pi/2 * i
            desiredx = 0.3 * math.cos(2*math.pi*frame/interval + offset) / (2*math.pi)
            desiredy = 0.3 * math.sin(2*math.pi*frame/interval + offset) / (2*math.pi)
            desiredz = 0.0 * math.sin(2*math.pi*frame/interval + offset) / (2*math.pi)
            animatedTargets[i] = np.add(animatedTargets[i], [desiredx, desiredy, desiredz]) if animatedTargets[i] is not None else None

        for i in range(animatedOrientations.shape[0]):
            desiredxrot = 1 * math.sin(2*math.pi*frame/interval + offset) * math.pi
            desiredyrot = 0 *math.sin(2*math.pi*frame/interval + offset) * math.pi
            desiredzrot = 0 * math.sin(2*math.pi*frame/interval + offset) * math.pi
            animatedOrientations[i] = [desiredxrot, desiredyrot, desiredzrot]

        # for i in range(animatedOrientations.shape[0]):
        #     animatedOrientations[i] = [2*math.pi*frame/interval for j in range(3)]

    iterations = 10
    skeleton = engine.getSkeleton(targets=animatedTargets, orientations=animatedOrientations) # Adjacency list of joint connections
    for i in range(iterations):
        skeleton = engine.getSkeleton(targets=animatedTargets, orientations=animatedOrientations) # Adjacency list of joint connections
    
    #print(desiredx - skeleton[-1][-1][0], desiredy - skeleton[-1][-1][1], desiredz - skeleton[-1][-1][2])
    # Wipe old plot
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    for i, adjacency in enumerate(skeleton):
        if len(adjacency) != 2:
            print("Incorrect element length")
        ax.plot([adjacency[0][0], adjacency[1][0]], [adjacency[0][1], adjacency[1][1]], [adjacency[0][2], adjacency[1][2]], color=get_colour(i))

def on_close(event):
    exit()

def main(model:str):
    global target_positions, target_orientations, engine
    joints = np.array([])
    links = np.array([])

    match model:
        case "0":
            # SSRT ARM
            links = np.array([[0.0, 0.0, 0.4, 0.2, 0.3, 0.1]])
            DH_params = np.array([[links[0][0], 0, 0, 0],
                                [links[0][1], 0, 0, 0],
                                [links[0][2], 0, 0, 0],
                                [links[0][3], 0, 0, 0],
                                [links[0][4], 0, 0, 0],
                                [links[0][5], 0, 0, 0]])
            joint_map = np.array([[0, 1], [1, 3], [2, 3], [3, 1], [4, 3], [5, 1]])

            # for [i,j] in joint_map:
            #     DH_params[i][j] = math.pi/4

            [i,j] = joint_map[0]
            DH_params[i,j] = math.pi/4

            [i,j] = joint_map[1]
            DH_params[i,j] = -math.pi/4

            [i,j] = joint_map[2]
            DH_params[i,j] = 2*math.pi/4

            [i,j] = joint_map[3]
            DH_params[i,j] = 0*math.pi/4

            [i,j] = joint_map[4]
            DH_params[i,j] = 0*math.pi/4

            [i,j] = joint_map[5]
            DH_params[i,j] = 0*math.pi/4



            target_positions = np.array([[0.3,0.3,0.3]])
            target_orientations = np.array([[0,0,0]])
            origin = np.array([[0, 0, 0]])

            animationEnabled = False
            manipulators = [SSRTArm(DH_params, joint_map, origin)]
            engine = IKEngine(manipulators)
        case "1":
            joints = np.array([[0,0,0]])
            links = np.array([[0.3,1,1]])
            target_positions = np.array([[0.3, -0.0001, -1.2]])
            origins = np.array([[0, 0, 0]])

            # joints = np.array([[0,0,0]])
            # links = np.array([[0.3,1,1]])
            # targets = np.array([[0.4, -0.4, -1.2]])
            # origins = np.array([[0.4,-0.2,0]])

            animationEnabled = True
            manipulators = [XOriented3DOF3LinkArm(links[i], joints[i], origins[i]) for i in range(len(joints))]
            engine = IKEngine(manipulators)
        case "2":
            joints = np.array([[0*math.pi/2,
                                0*math.pi/2,
                                0*math.pi/2, 
                                0*math.pi/2, 
                                0*math.pi/2, 
                                0*math.pi/2]])
            
            links = np.array([[0.4,0.3, 0.2]])
            animationEnabled = False
            manipulators = [SSRTArmWrapper(links[0], joints[0], [], [])]
            engine = IKEngine(manipulators)
        
        case _:
            joints = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
            links = np.array([[0.3,1,1], [0.3,1,1], [0.3,1,1], [0.3,1,1]])
            target_positions = np.array([[0.4, -0.4, -1.2], [-0.2, -0.4, -1.2], [0.4, 0.4, -1.2], [-0.2, 0.4, -1.2]])
            origins = np.array([[0.4,-0.2,0], [-0.4,-0.2,0], [0.4,0.2,0], [-0.4,0.2,0]])

            # joints = np.array([[0,0,0]])
            # links = np.array([[0.3,1,1]])
            # targets = np.array([[0.4, -0.4, -1.2]])
            # origins = np.array([[0.4,-0.2,0]])

            animationEnabled = True
            manipulators = [XOriented3DOF3LinkArm(links[i], joints[i], origins[i]) for i in range(len(joints))]
            engine = IKEngine(manipulators)
    
    # Validate targets
    maxmag = 0
    for i in range(min(len(links), len(target_positions))):
        if target_positions[i] is not None:
            maxmag = max(maxmag, np.sum(links[i]))
            if (magnitude(target_positions[i]) > maxmag):
                print("Position impossible!")
                return False
    
    if maxmag == 0:
        maxmag = 1

    ticks_frequency = 0.5
    # ax.plot([0, l1], [0, 0], [0, 0], [0, l2], marker='o') I don't remember what this did

    ax.set_aspect('equal', adjustable='box')


    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=0, x=1.2*maxmag)
    ax.set_ylabel('y', size=14, labelpad=0, y=1.2*maxmag, rotation=0)
    ax.set_zlabel('z', size=14, labelpad=0, y=1.2*maxmag, rotation=0)

    # Create custom major ticks to determine position of tick labels
    ticks = np.arange(-maxmag, maxmag+1, ticks_frequency)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(-maxmag, maxmag+1), minor=True)
    ax.set_yticks(np.arange(-maxmag, maxmag+1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    ax.set_xlim(-maxmag, maxmag)
    ax.set_ylim(-maxmag, maxmag)
    ax.set_zlim(-maxmag, maxmag)

    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect("key_press_event", key_handler)

    ani = animation.FuncAnimation(fig=fig, func=functools.partial(updatePlot, animationEnabled=animationEnabled), frames=frames, interval=interval, repeat= not save_animation)

    if save_animation:
        filepath = "ssrt_position.gif" 
        writergif = animation.PillowWriter(fps=1/(interval/1000))
        ani.save(filepath, writer=writergif)
    else:
        plt.show() 


if __name__ == "__main__":
    model = None
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    main(model)