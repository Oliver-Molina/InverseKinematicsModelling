import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

l1 = 1
l2 = 1

x = 0.7
y = 0.7

frames = 500
interval = 50

fig = plt.figure()
ax = fig.add_subplot()
lines = ax.plot([0, l1], [0, 0], [0, 0], [0, l2], marker='o')

def generate_arm_angles(x, y, l1, l2):
    theta2 = math.acos((x**2 + y**2 - l2**2 - l1**2)/(2*l2*l1))
    theta1 = math.atan(y/x) - math.atan((l2*math.sin(theta2))/(l2*math.cos(theta2) + l1))
    return [theta1, theta2]


def magnitude(a, b):
    return math.sqrt(a**2 + b**2)

def update2dArm(frame):
    # Plot Data
    desiredx = x + 0.5 * math.cos(2*math.pi*frame/interval)
    desiredy = y + 0.5 * math.sin(2*math.pi*frame/interval)

    theta1, theta2 = generate_arm_angles(desiredx, desiredy, l1 , l2)

    x1 = l1*math.cos(theta1)
    y1 = l1*math.sin(theta1)

    x2 = x1 + l2*math.cos(theta2 + theta1)
    y2 = y1 + l2*math.sin(theta2 + theta1)

    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    ax.plot([0, x1], [0, y1], [x1, x2], [y1, y2], color='b', marker='o')

def main():
    maxmag = l1 + l2
    ticks_frequency = 0.25

    if (magnitude(x,y) > maxmag):
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
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(-maxmag, maxmag+1, ticks_frequency)
    y_ticks = np.arange(-maxmag, maxmag+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

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

    # ax.set_xlim(-maxmag, maxmag)
    # ax.set_ylim(-maxmag, maxmag)
    # ax.set_zlim(-maxmag, maxmag)

    plt.axis([-maxmag, maxmag,-maxmag, maxmag])

    ani = animation.FuncAnimation(fig=fig, func=update2dArm, frames=frames, interval=interval, repeat=False)

    plt.show()







if __name__ == "__main__":
    main()