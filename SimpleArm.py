import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

l1 = 1
l2 = 1

x = 0.5
y = 0.7
z = 0.7

frames = 5000
interval = 50

save_animation = False

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lines = ax.plot([0, l1], [0, 0], [0, 0], [0, l2], marker='o')


def magnitude(x):
    return math.sqrt(sum(i**2 for i in x))

def get_colour(i):
    colours = ['b', 'g', 'r', 'c', 'm', 'y']
    return colours[i % len(colours)]

def generate2JointSinglePlainArm(x, y, l1, l2):
    theta2 = math.acos((x**2 + y**2 - l2**2 - l1**2)/(2*l2*l1))
    theta1 = math.atan(y/x) - math.atan((l2*math.sin(theta2))/(l2*math.cos(theta2) + l1))
    return [theta1, theta2]

def update2JointSinglePlainArm(frame):
    # Plot Dataupdate2JointSinglePlainArm
    desiredx = x + 0.5 * math.cos(2*math.pi*frame/interval)
    desiredy = y + 0.5 * math.sin(2*math.pi*frame/interval)

    theta1, theta2 = generate2JointSinglePlainArm(desiredx, desiredy, l1 , l2)

    x1 = l1*math.cos(theta1)
    y1 = l1*math.sin(theta1)

    x2 = x1 + l2*math.cos(theta2 + theta1)
    y2 = y1 + l2*math.sin(theta2 + theta1)

    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    ax.plot([0, x1], [0, y1], [0,0], color='b', marker='o')
    ax.plot([x1, x2], [y1, y2], [0,0], color='r', marker='o')

def generate3Joint3DArmAnglesZ(x, y, z, l1, l2):
    theta3 = math.acos((x**2 + y**2 + z**2 - l2**2 - l1**2)/(2*l2*l1))
    theta2 = math.atan(z/math.sqrt(x**2 + y**2)) - math.atan((l2*math.sin(theta3))/(l2*math.cos(theta3) + l1))
    theta1 = math.atan(y/x)
    return [theta1, theta2, theta3]

def update3Joint3DArmZ(frame):
    # Animate Initial Values
    desiredx = x + 0.5 * math.cos(2*math.pi*frame/interval)
    desiredy = y + 0.5 * math.sin(2*math.pi*frame/interval)
    desiredz = z + 0.5 * math.sin(2*math.pi*frame/interval)

    # Generate Angles
    theta1, theta2, theta3 = generate3Joint3DArmAnglesZ(desiredx, desiredy, desiredz, l1 , l2)

    # Compute all positions using forward Kinematics

    P0 = np.array([0, 0, 0])

    r1 = l1*math.cos(theta2)

    P1 = np.add(P0, [r1*math.cos(theta1), r1*math.sin(theta1), l1*math.sin(theta2)])

    r2 = l2*math.cos(theta2 + theta3)

    P2 = np.add(P1, [r2*math.cos(theta1), r2*math.sin(theta1), l2*math.sin(theta2 + theta3)])

    # Wipe old plot
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    # Plot lines connecting each point (origin, P1), (P1, P2)
    # ax.plot([0, x1], [0, y1], [0,0], color='b', marker='o')
    # ax.plot([x1, x2], [y1, y2], [0,0], color='r', marker='o')

    P = [P0, P1, P2]
    if len(P) < 2:
        return
    
    for i in range(1, len(P)):
        ax.plot([P[i - 1][0], P[i][0]], [P[i - 1][1], P[i][1]], [P[i - 1][2], P[i][2]], color=get_colour(i))


def generate3Joint3DArmAnglesX(x, y, z, l1, l2):
    theta3 = math.acos((x**2 + y**2 + z**2 - l2**2 - l1**2)/(2*l2*l1))
    theta2 = math.atan(x/math.sqrt(y**2 + z**2)) - math.atan((l2*math.sin(theta3))/(l2*math.cos(theta3) + l1))
    theta1 = math.atan(z/y)
    return [theta1, theta2, theta3]

def update3Joint3DArmX(frame):
    # Animate Initial Values
    desiredx = x + 0.5 * math.cos(2*math.pi*frame/interval)
    desiredy = y + 0.5 * math.sin(2*math.pi*frame/interval)
    desiredz = z + 0.5 * math.sin(2*math.pi*frame/interval)

    # Generate Angles
    theta1, theta2, theta3 = generate3Joint3DArmAnglesX(desiredx, desiredy, desiredz, l1 , l2)

    # Compute all positions using forward Kinematics

    P0 = np.array([0, 0, 0])

    r1 = l1*math.cos(theta2)

    P1 = np.add(P0, [l1*math.sin(theta2), r1*math.sin(theta1), r1*math.cos(theta1)])

    r2 = l2*math.cos(theta2 + theta3)

    P2 = np.add(P1, [l2*math.sin(theta2 + theta3), r2*math.sin(theta1), r2*math.cos(theta1)])

    # Wipe old plot
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    # Plot lines connecting each point (origin, P1), (P1, P2)
    # ax.plot([0, x1], [0, y1], [0,0], color='b', marker='o')
    # ax.plot([x1, x2], [y1, y2], [0,0], color='r', marker='o')

    P = [P0, P1, P2]
    if len(P) < 2:
        return
    
    for i in range(1, len(P)):
        ax.plot([P[i - 1][0], P[i][0]], [P[i - 1][1], P[i][1]], [P[i - 1][2], P[i][2]], color=get_colour(i))

def main():
    maxmag = l1 + l2
    ticks_frequency = 0.5

    if (magnitude([x,y]) > maxmag):
        print("Position impossible!")
        return False

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


    ani = animation.FuncAnimation(fig=fig, func=update3Joint3DArmX, frames=frames, interval=interval, repeat=False)

    if save_animation:
        filepath = "animation.gif" 
        writergif = animation.PillowWriter(fps=1/(interval/1000)) 
        ani.save(filepath, writer=writergif)
    else:
        plt.show()







if __name__ == "__main__":
    main()