from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib import animation
import random

fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto')
v = 300
q = 9.0 
A = np.square(v) / q
omega = q / (2 * v)
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

def f_k_k_1(od, t):
    return np.array([ [1, 0, t       , 0        , 0.5*t*t      , 0            ],
                      [0, 1, 0       , t        , 0            , 0.5*t*t      ],
                      [0, 0, 1       , 0        , t            , 0            ],
                      [0, 0, 0       , 1        , 0            , t            ],
                      [0, 0, 0       , 0        , 1            , 0            ],
                      [0, 0, 0       , 0        , 0            , 1            ]])
class kalman_data:
    def __init__(self, mu, covMat):
        self.mu = mu
        self.covMat = covMat
    def __str__(self):
        return str(self.mu) + "\n" + str(self.covMat)

def d_k_k_1(sigma, dt):
    return (sigma*sigma) * np.array([ 
        [ 0.25 * (dt**4)  , 0              , 0.5 * (dt**3), 0            , 0.5*dt*dt     , 0        ] ,
        [ 0               , 0.25 * (dt**4)  , 0           , 0.5 * (dt**3), 0             , 0.5*dt*dt] ,
        [ 0.5 * (dt**3)   , 0              , (dt**2)      , 0            , dt            , 0        ] ,
        [ 0               , 0.5 * (dt**3)   , 0           , (dt**2)      , 0             , dt       ] ,
        [ 0.5 * (dt**2)   , 0              , dt           , 0            , 1             , 0        ] ,
        [ 0               , 0.5 * (dt**2)   , 0           , dt           , 0             , 1        ] ])

 
class radar:
    def __init__(self, x_position, y_position, arc_error, distance_error, no_data, R):
        self.x_position = x_position
        self.y_position =  y_position
        self.arc_error = arc_error
        self.distance_error = distance_error
        self.no_data = no_data
        self.last_positions = []
        self.R = R

    def next_data(self, x, y):
        [distance, arc] = cartesian_to_polar(x-self.x_position, y-self.y_position)
        distance = distance + (2 * np.random.random_sample() - 1) * self.distance_error
        arc = arc + (2 * np.random.random_sample() - 1) * self.arc_error
        if (not self.no_data) or np.random.randint(2) == 1:
            data = polar_to_cartesian(distance, arc)
            x = data[0] + self.x_position
            y = data[1] + self.y_position
            self.last_positions.append([x , y])
        else:
            self.last_positions.append(None)

    def get_data(self , t):
        return self.last_positions[t]

    def get_polar_data(self , t):
        data = self.get_data(t)
        if data == None:
            return None
        return cartesian_to_polar(data[0]-self.x_position, data[1]-self.y_position)

    def show_point(self, t):
        data = self.get_polar_data(t)
        ax.plot([self.last_positions[-1][0]],
                [self.last_positions[-1][1]] , '.', color='yellow')

    def show_bahn(self):
        ax.plot(self.last_positions[0], self.last_positions[1], label='Flugbahn von radar sicht',color='purple', lw=0.5)  # Flugbahn von radar sicht

    def show_ellipse(self, t):
        data = self.get_polar_data(t)
        ax.add_patch(Ellipse((self.last_positions[-1][0], self.last_positions[-1][1]), 300, 60*data[0] * self.arc_error, data[1]*180/np.pi))

    def show_sensor(self):
        ax.plot([self.x_position] , [self.y_position], 'x' , color='red')
     
class kalman_filter:
    def __init__(self, object_data, t):
        self.object_data = [[object_data, d_k_k_1(16, 1), t]]

    def prediction(self, xk_1k_1, pk_1k_1, sigma ,dt):
        dkk_1 = d_k_k_1(sigma, dt)
        fkk_1 = f_k_k_1(xk_1k_1, dt)
        xkk_1 = fkk_1.dot(xk_1k_1)
        pkk_1 = fkk_1.dot(pk_1k_1).dot(np.transpose(fkk_1)) + dkk_1
        return [xkk_1, pkk_1]
        

    def filtering(self, sensorData, k, Hk, Rk,sigma, t):
        muk_1 = self.object_data[k-1][0]
        covMatk_1 = self.object_data[k-1][1]
        tk_1 = self.object_data[k-1][2]
        [mukk_1, covMatkk_1] = self.prediction(muk_1, covMatk_1 ,sigma ,t - tk_1)

        zkk = np.array([sensorData[0],sensorData[1]])
        vkk_1 = zkk - Hk.dot(muk_1)
        skk_1 = Hk.dot(covMatkk_1).dot(np.transpose(Hk)) + Rk
        wkk_1 = covMatkk_1.dot(np.transpose(Hk).dot(np.linalg.inv(skk_1)))
        xkk = mukk_1 + wkk_1.dot(vkk_1)
        covMatkk = covMatkk_1 - wkk_1.dot(skk_1).dot(np.transpose(wkk_1))
        self.object_data.append([xkk , covMatkk ,  t])
    
    def render(self, t):
        mu = self.object_data[t][0]
        ax.plot([mu[0]], [mu[1]], '.', color="orange")

    
radars = [
    radar(-10000, -5000, 0.001, 50, False, np.array([[2500, 0] , [0, 2500]])),
    radar( 10000, -5000, 0.001, 200, False, np.array([[2500, 0] , [0, 2500]])),
    radar(     0,  5000, 0.1, 50, False, np.array([[2500, 0] , [0, 2500]]))
]

od = np.array([x[0],y[0],velocity_x[0], velocity_y[0],accel_x[0],accel_y[0]])  
kal = kalman_filter(od, 0)
Hk = np.array([[1,0,0,0,0, 0],
               [0,1,0,0,0, 0]])

def animate(t):
    i = t % len(x)
    ax.clear()
    # ax.set_xlim(-11000, 11000)
    # ax.set_ylim(-11000, 11000)
    for radar in radars:
        radar.next_data(x[i], y[i] )
    [Rk , zk]= multiSensor_rechner( radars, t)
    print(zk - np.array(radars[0].get_data(t)))
    # zk = np.array([x[i],y[i]]) 
    if t > 0:
        kal.filtering(zk, t, Hk, Rk, 10, t)
        kal.render(t)
    show_flugbahn(t)
    for radar in radars:
        radar.show_sensor()
        # radar.show_point(t)
        # radar.show_ellipse(t)
    
    # 
    # show_velocity(t, True)
    # show_accel(t, True)
    # radar1.show_point(x[i], y[i], i)
    # radar1.show_bahn()



def show_flugbahn(t):
    i = t % len(x)
    ax.plot(x, y, label='Flugbahn', color='black', lw=0.5)  # Flugbahn
    ax.plot([x[i]], [y[i]], '.')  # Flugzeugposition


def show_velocity(t, dim_2d):
    i = t % len(x)
    if dim_2d == False:
        ax.plot([x[i], x[i] + 5*velocity_x[i]],
                [y[i], y[i]], [1.0, 1.0], color='blue')
        ax.plot([x[i], x[i]], [y[i], y[i] + 5*velocity_y[i]], color='red')
    else:
        ax.plot([x[i], x[i] + 5*velocity_x[i]],
                [y[i], y[i] + 5*velocity_y[i]], color='blue')


def show_accel(t, dim_2d):
    i = t % len(x)
    if dim_2d == False:
        ax.plot([x[i], x[i] + 100*accel_x[i]],
                [y[i], y[i]], color='red')
        ax.plot([x[i], x[i]], [y[i], y[i] + 100*accel_y[i]],
                 color='orange')
    else:
        ax.plot([x[i], x[i] + 100*accel_x[i]],
                [y[i], y[i] + 100*accel_y[i]], color='green')


def polar_to_cartesian(r, phi):
    x = np.cos(phi) * r
    y = np.sin(phi) * r
    return [x, y]

def cartesian_to_polar(x , y):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    phi = np.arctan2(y, x)
    return [r , phi]

def N(x, mu, covMat):
    return 1/np.sqrt((2*np.pi)**2 * np.linalg.det(covMat)) * np.exp(-0.5 * ((np.transpose(x - mu)).dot(np.linalg.inv(covMat))).dot(x - mu))

def multiSensor_rechner(radars , i):
    Rksum = np.array([[0.0,0.0],[0.0, 0.0]])
    RK_inv_sum = np.array([[0.0,0.0],[0.0, 0.0]])
    zksum = np.array([0.0, 0.0])
    for radar in radars:
        R_inv = np.linalg.inv(radar.R)
        RK_inv_sum = RK_inv_sum + R_inv
        zksum = zksum + R_inv.dot(np.array(radar.get_data(i)))
        Rksum = Rksum + radar.R
    Rk = np.linalg.inv(RK_inv_sum)   #harmonic avg
    zk = Rk.dot(zksum)
    Rk = Rksum / len(radars) #avg
    return [Rk , zk]

        

ani=animation.FuncAnimation(fig, animate, interval=100)

plt.show()


