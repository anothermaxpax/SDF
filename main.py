from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib import animation
import pprint
import random

fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto')
v = 300
q = 9
A = np.square(v) / q
omega = q / 2 / v
print("A = ", A)
print("omega = ", omega)
sensor_position = [0, 0]
sensor_radial_error = 2
sensor_distance_error = 10

t = np.arange(0, 2.01*np.pi/omega, 1)
x = A * np.sin(omega * t)
y = A * np.sin(2.0 * omega * t)
velocity_x = v * np.cos(omega * t)
velocity_y = v * np.cos(2 * omega * t)
accel_x = -q * np.sin(omega * t)/4
accel_y = -q * np.sin(2 * omega * t)

class radar:
    def __init__(self, x_position, y_position, arc_error, distance_error, no_data):
        self.x_position = x_position
        self.y_position = y_position
        self.arc_error = arc_error
        self.distance_error = distance_error
        self.no_data = no_data
        self.last_positions = [[], []]

    def get_data(self, x, y, t):
        distance = np.sqrt(np.power(x - self.x_position, 2) + np.power(y - self.y_position, 2)
                           ) + (2 * np.random.random_sample() - 1) * self.distance_error
        arc = np.arctan2(y - self.y_position, x - self.x_position) + (2 * np.random.random_sample() - 1) * self.arc_error
        if (not self.no_data) or np.random.randint(2) == 1:
            data = polar_to_normal(distance, arc)
            x = data[0] + self.x_position
            y = data[1] + self.y_position
            self.last_positions[0].append(x)
            self.last_positions[1].append(y)
            return [distance, arc]
        return None

    def show_point(self, x, y, t):
        data = self.get_data(x, y, t)
        ax.plot([self.last_positions[0][-1]],
                [self.last_positions[1][-1]] , '.', color='yellow')

        ax.add_patch(Ellipse((self.last_positions[0][-1], self.last_positions[1][-1]), 300, 6*data[0] * self.arc_error, data[1]*180/np.pi))

    def show_bahn(self):
        ax.plot(self.last_positions[0], self.last_positions[1], label='Flugbahn von radar sicht',
                color='purple', lw=0.5)  # Flugbahn von radar sicht


radar1 = radar(0, 0, 0.03, 50, False)


def animate(i):
    ax.clear()
    ax.set_xlim(-11000, 11000)
    ax.set_ylim(-11000, 11000)
    show_flugbahn(i)
    show_velocity(i, True)
    show_accel(i, True)
    radar1.show_point(x[i], y[i], i)
    radar1.show_bahn()


def show_flugbahn(i):
    ax.plot(x, y, label='Flugbahn', color='black', lw=0.5)  # Flugbahn
    ax.plot([x[i]], [y[i]], '.')  # Flugzeugposition


def show_velocity(i, dim_2d):
    if dim_2d == False:
        ax.plot([x[i], x[i] + 5*velocity_x[i]],
                [y[i], y[i]], [1.0, 1.0], color='blue')
        ax.plot([x[i], x[i]], [y[i], y[i] + 5*velocity_y[i]], color='red')
    else:
        ax.plot([x[i], x[i] + 5*velocity_x[i]],
                [y[i], y[i] + 5*velocity_y[i]], color='blue')


def show_accel(i, dim_2d):

    if dim_2d == False:
        ax.plot([x[i], x[i] + 100*accel_x[i]],
                [y[i], y[i]], color='red')
        ax.plot([x[i], x[i]], [y[i], y[i] + 100*accel_y[i]],
                 color='orange')
    else:
        ax.plot([x[i], x[i] + 100*accel_x[i]],
                [y[i], y[i] + 100*accel_y[i]], color='green')


def polar_to_normal(r, phi):
    x = np.cos(phi) * r
    y = np.sin(phi) * r
    return [x, y]


def N(x, mu, covMat):
    return 1/np.sqrt((2*np.pi)**2 * np.linalg.det(covMat)) * np.exp(-0.5 * ((np.transpose(x - mu)).dot(np.linalg.inv(covMat))).dot(x - mu))


ani=animation.FuncAnimation(fig, animate, interval=100)

plt.show()
