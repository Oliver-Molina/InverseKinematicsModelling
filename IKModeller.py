import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import functools
from IKEngine import *
from CanisArmModel import *
import sys

frames = 100
interval = 50

save_animation = False

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def magnitude(x):
    return math.sqrt(sum(i**2 for i in x))

def get_colour(i):
    colours = ['b', 'g', 'r', 'c', 'm', 'y']
    return colours[i % len(colours)]

# def getCanisSkeleton(frame):
#     # Animate Initial Values

#     NUM_OF_LEGS = 4

#     skeleton = np.array([[]])

#     BODY_WIDTH = 1

#     BODY_LENGTH = 2.5

#     leg_origins = np.array([[BODY_LENGTH/2,BODY_WIDTH/2, 0], 
#                             [BODY_LENGTH/2,-BODY_WIDTH/2, 0], 
#                             [-BODY_LENGTH/2,BODY_WIDTH/2, 0], 
#                             [-BODY_LENGTH/2,-BODY_WIDTH/2, 0]])
    
#     # add frame of body to skeleton
#     skeleton = np.append(skeleton, [leg_origins[0], leg_origins[1]])
#     skeleton = np.append(skeleton, [leg_origins[1], leg_origins[3]])
#     skeleton = np.append(skeleton, [leg_origins[3], leg_origins[2]])
#     skeleton = np.append(skeleton, [leg_origins[2], leg_origins[0]])

#     for i in range(NUM_OF_LEGS):
#         desiredx = x + 0.5 * math.cos(2*math.pi*frame/interval - i*math.pi/NUM_OF_LEGS)
#         desiredy = 1 #y if i % 2 == 0 else -y #+ 0.5 * math.sin(2*math.pi*frame/interval)
#         desiredz = z + 0.5 * math.sin(2*math.pi*frame/interval - i*math.pi/NUM_OF_LEGS)

#         position = np.array([desiredx, desiredy, desiredz])

#         leg_skeleton = update3Joint3DArmX(frame, position)

#         # translate each set of joints to the leg origin
#         for j in range(len(leg_skeleton)):
#             for k in range(len(leg_skeleton[j])):
#                 leg_skeleton[j][k] = np.add(leg_skeleton[j][k], leg_origins[i])
        

#         # add legs to skeleton
#         skeleton = np.append(skeleton, leg_skeleton)

#     skeleton = skeleton.reshape(-1, 2, 3)
#     return skeleton

# def getCanisSkeleton(frame):
#     # Animate Initial Values

#     NUM_OF_LEGS = 4

#     skeleton = np.array([[]])

#     BODY_WIDTH = 1

#     BODY_LENGTH = 2.5

#     leg_origins = np.array([[BODY_LENGTH/2,BODY_WIDTH/2, 0], 
#                             [BODY_LENGTH/2,-BODY_WIDTH/2, 0], 
#                             [-BODY_LENGTH/2,BODY_WIDTH/2, 0], 
#                             [-BODY_LENGTH/2,-BODY_WIDTH/2, 0]])
    
#     # add frame of body to skeleton
#     skeleton = np.append(skeleton, [leg_origins[0], leg_origins[1]])
#     skeleton = np.append(skeleton, [leg_origins[1], leg_origins[3]])
#     skeleton = np.append(skeleton, [leg_origins[3], leg_origins[2]])
#     skeleton = np.append(skeleton, [leg_origins[2], leg_origins[0]])

#     for i in range(NUM_OF_LEGS):
#         desiredx = x + 0.5 * math.cos(2*math.pi*frame/interval - i*math.pi/NUM_OF_LEGS)
#         desiredy = 1 #y if i % 2 == 0 else -y #+ 0.5 * math.sin(2*math.pi*frame/interval)
#         desiredz = z + 0.5 * math.sin(2*math.pi*frame/interval - i*math.pi/NUM_OF_LEGS)

#         position = np.array([desiredx, desiredy, desiredz])

#         leg_skeleton = update3Joint3DArmX(frame, position)

#         # translate each set of joints to the leg origin
#         for j in range(len(leg_skeleton)):
#             for k in range(len(leg_skeleton[j])):
#                 leg_skeleton[j][k] = np.add(leg_skeleton[j][k], leg_origins[i])
        

#         # add legs to skeleton
#         skeleton = np.append(skeleton, leg_skeleton)

#     skeleton = skeleton.reshape(-1, 2, 3)
#     return skeleton

def updatePlot(frame, ikEngine, targets, animationEnabled):
    animatedTargets = targets
    if animationEnabled:
        for i in range(animatedTargets.shape[0]):
            offset = math.pi/2 * i
            desiredx = 0.3 * math.cos(2*math.pi*frame/interval + offset) / (2*math.pi)
            desiredy = 0.0 * math.sin(2*math.pi*frame/interval + offset) / (2*math.pi)
            desiredz = 0.2 * math.sin(2*math.pi*frame/interval + offset) / (2*math.pi)
            animatedTargets[i] = np.add(animatedTargets[i], [desiredx, desiredy, desiredz])

    skeleton = ikEngine.getSkeleton(targets=animatedTargets) # Adjacency list of joint connections

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
    joints = np.array([])
    links = np.array([])
    targets = np.array([])
    engine = None

    match model:
        case _:
            joints = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
            links = np.array([[0.3,1,1], [0.3,1,1], [0.3,1,1], [0.3,1,1]])
            targets = np.array([[0.4, -0.4, -1.2], [-0.2, -0.4, -1.2], [0.4, 0.4, -1.2], [-0.2, 0.4, -1.2]])
            origins = np.array([[0.4,-0.2,0], [-0.4,-0.2,0], [0.4,0.2,0], [-0.4,0.2,0]])
            animationEnabled = True
            manipulators = [XOriented3DOF3LinkArm(links[i], joints[i], origins[i]) for i in range(len(joints))]
            engine = AnalyticalIKEngine(manipulators)
    
    # Validate targets
    maxmag = 0
    for i in range(min(len(links), len(targets))):
        maxmag = max(maxmag, np.sum(links[i]))
        if (magnitude(targets[i]) > maxmag):
            print("Position impossible!")
            return False
    
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

    ani = animation.FuncAnimation(fig=fig, func=functools.partial(updatePlot, ikEngine=engine, targets=targets, animationEnabled=animationEnabled), frames=frames, interval=interval, repeat=False)

    if save_animation:
        filepath = "animation.gif" 
        writergif = animation.PillowWriter(fps=1/(interval/1000))
        ani.save(filepath, writer=writergif)
    else:
        plt.show() 







if __name__ == "__main__":
    model = None
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    main(model)