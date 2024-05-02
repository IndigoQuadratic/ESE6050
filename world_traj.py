from typing import Any, Union

import numpy as np
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from scipy.optimize import minimize
from .graph_search import graph_search
from .occupancy_map import OccupancyMap

class WorldTraj(object):
    """
    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """
        # over_under 14 s ; maze 10.11 s
        # self.resolution = np.array([0.22, 0.22, 0.22])
        # self.margin = 0.7
        # self.mid_eps = 1
        # self.rdp_eps = 0.8
        # self.v = 2.51

        #
        # self.resolution = np.array([0.22, 0.22, 0.22])
        # self.margin = 0.67
        # self.mid_eps = 1
        # self.rdp_eps = 0.25 # 0.25
        # self.v = 2.6 # 2.6

        self.resolution = np.array([0.235, 0.235, 0.235])
        self.margin = 0.48
        self.mid_eps = 0.5
        self.rdp_eps = 0.25 # 0.25
        self.v = 3.7 # 2.6 3.1 3.75 over_under collides // 3.2 OVER_UNDER DOESN'T COLLIDE
        self.start = np.array(start)
        self.goal = np.array(goal)

        occ_map = OccupancyMap(world, self.resolution, self.margin)

        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        self.points = np.zeros((1, 3))  # shape=(n_pts,3)


        # STUDENT CODE HERE
        self.points = np.array(RDP_points(self.path, epsilon=self.rdp_eps))
        self.insert_midpoints(epsilon=self.mid_eps)
        self.insert_intermediate_points()
        self.angles = self.calculate_angles()
        self.Time_segment()
        self.cal_A()
        self.cal_bs()
        self.cal_H()


        self.cons_x = [
            {'type': 'eq', 'fun': self.linear_constraint_x},
        ]
        self.cons_y = [
            {'type': 'eq', 'fun': self.linear_constraint_y},
        ]
        self.cons_z = [
            {'type': 'eq', 'fun': self.linear_constraint_z},
        ]

        x0 = np.ones((6 * len(self.ts)))
        self.coefficient_x = minimize(self.cost_func, x0, constraints=self.cons_x, method="SLSQP")
        self.Cx = self.coefficient_x.x
        print("x optimization sucess: ", self.coefficient_x.success)
        self.coefficient_y = minimize(self.cost_func, x0, constraints=self.cons_y, method="SLSQP")
        self.Cy = self.coefficient_y.x
        print("y optimization sucess: ", self.coefficient_y.success)
        self.coefficient_z = minimize(self.cost_func, x0, constraints=self.cons_z, method="SLSQP")
        self.Cz = self.coefficient_z.x
        print("z optimization sucess: ", self.coefficient_z.success)

    def ver_dist(A, B, C):  # calculate the perpendicular distance between point C and segment AB
        AB = B - A
        AC = C - A
        cross_vec = np.cross(AB, AC)
        area = np.linalg.norm(cross_vec)
        len_AB = np.linalg.norm(AB)
        if len_AB == 0:
            return np.linalg.norm(AC)
        else:
            dist = area / len_AB
            return dist

    def path_to_points(self):
        distance_threshold = np.linalg.norm(self.start - self.goal) / 6

        if self.path.any():
            self.points = [self.path[0]]
            last_vector = self.path[1] - self.path[0]

            for i in range(2, len(self.path)):
                current_point = self.path[i]
                current_vector = current_point - self.path[i - 1]


                if not np.allclose(current_vector, last_vector):
                    self.points.append(self.path[i - 1])

                elif np.linalg.norm(current_point - self.points[-1]) > distance_threshold:
                    self.points.append(self.path[i - 1])

                last_vector = current_vector

            self.points.append(self.path[-1])

    def Time_segment(self):
        # Calculate Euclidean distances between consecutive points
        self.Ti = np.linalg.norm(np.diff(self.points, axis=0), axis=1)

        # Apply a special scaling factor to the first and last time segments
        self.Ti[0] *= 1.4
        self.Ti[-1] *= 1.4

        # Adjust all time segments by applying a scaling conversion factor
        # self.Ti = 1.39 * (self.Ti ** (1/2))  # Operate on the entire array, not on individual elements
        self.Ti /= self.v
        # for i in range(0, self.Ti.shape[0] - 1):
        #     if 0 < self.angles[i] <= 90:
        #         self.Ti[i+1] *= 1.2
        #     # elif 45 < self.angles[i] <= 90:
        #     #     self.Ti[i+1] *= 1.075
        #     # elif 90 < self.angles[i] <= 150:
        #     #     self.Ti[i+1] *= 1.05
        #     elif 90 < self.angles[i] <= 160:
        #         self.Ti[i+1] *= 1.1
        # Compute cumulative timestamps, including an initial moment at 0
        self.start_true_ts = np.insert(np.cumsum(self.Ti), 0, 0)

        # Create a timestamps array matching the number of points
        self.ts =  np.ones(len(self.points) - 1)  # Use len(self.points) to get the number of points
        #self.ts = self.Ti
        self.start_t = np.insert(np.cumsum(self.ts), 0, 0)  # Cumulatively sum and insert a 0 at the start

    def insert_midpoints(self, epsilon):
        mid_idx = []
        mid_pts = []
        for i in range(1, len(self.points)):
            start_point = self.points[i - 1]
            end_point = self.points[i]
            distance = np.linalg.norm(end_point - start_point)
            if distance > 2 * epsilon:
                interp_points = np.linspace(start_point, end_point, num=4)
                mid_pts.extend(interp_points[1:3])
                mid_idx.extend([i, i])

            elif distance > epsilon:
                mid_point = (start_point + end_point) / 2
                mid_pts.append(mid_point)
                mid_idx.append(i)
        if mid_idx:
            self.points = np.insert(self.points, mid_idx, mid_pts, axis=0)

    def insert_intermediate_points(self):
        new_points = [self.points[0]]

        for i in range(1, len(self.points) - 1):
            p1, p2, p3 = self.points[i - 1], self.points[i], self.points[i + 1]
            v1 = p2 - p1
            v2 = p3 - p2
            angle = np.degrees(
                np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))

            if angle < 90:

                mid_point1 = p1 + (p2 - p1) / 2
                mid_point2 = p2 + (p3 - p2) / 2
                if not np.array_equal(mid_point1, new_points[-1]):
                    new_points.append(mid_point1)
                new_points.append(p2)
                if not np.array_equal(mid_point2, p2):
                    new_points.append(mid_point2)
            elif 90 <= angle < 180:

                dist1 = np.linalg.norm(v1)
                dist2 = np.linalg.norm(v2)
                if dist1 > dist2:
                    mid_point = p1 + (p2 - p1) / 2
                    if not np.array_equal(mid_point, new_points[-1]):
                        new_points.append(mid_point)
                else:
                    mid_point = p2 + (p3 - p2) / 2
                    if not np.array_equal(mid_point, p2):
                        new_points.append(mid_point)
                new_points.append(p2)
            else:

                if not np.array_equal(p2, new_points[-1]):
                    new_points.append(p2)
    def calculate_angles(self):
        angles = []
        for i in range(1, len(self.points) - 1):
            v1 = self.points[i] - self.points[i - 1]
            v2 = self.points[i + 1] - self.points[i]
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            angle_degrees = np.degrees(angle)  # degree
            angles.append(angle_degrees)
        return angles
    def cal_bs(self):
        k = len(self.Ti)
        self.bs = np.zeros((6 + 4 * (k - 1), 3))

        # Setting p1(0) and pk(tk)
        self.bs[0] = self.points[0]
        self.bs[3] = self.points[-1]

        # Setting Interm Points
        for i in range(k - 1):
            j = 2 * i + 6
            self.bs[j] = self.points[i + 1]
            self.bs[j + 1] = self.points[i + 1]

    def cal_A(self):
        k = len(self.ts)
        self.A = np.zeros(((6 + 4 * (k - 1), 6 * k)))
        self.A[:3, :6] = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0]
        ])
        tk = self.ts[-1]
        self.A[3:6, -6:] = np.array([
            [tk ** 5, tk ** 4, tk ** 3, tk ** 2, tk ** 1, 1],
            [5 * tk ** 4, 4 * tk ** 3, 3 * tk ** 2, 2 * tk ** 1, 1, 0],
            [20 * tk ** 3, 12 * tk ** 2, 6 * tk ** 1, 2, 0, 0]
        ])
        for i in range(k - 1):
            ti = self.ts[i]
            self.A[2*i + 2*(k-1) + 6:2*i + 2*(k-1) + 8, 6*i:6*i+12] = np.array([
                  [5*ti ** 4, 4*ti ** 3, 3*ti ** 2, 2*ti ** 1, 1, 0, 0, 0, 0, 0, -1, 0],
                  [20*ti ** 3, 12*ti ** 2, 6*ti, 2, 0, 0, 0, 0, 0, -2, 0, 0],
            ])

        for i in range(k - 1):
            ti = self.ts[i]
            self.A[2*i + 6, 6*i:6*i+6] = np.array([ti ** 5, ti ** 4, ti ** 3, ti ** 2, ti ** 1, 1])
            self.A[2*i + 7, 6*(i+1)+5] = 1

    def cal_H(self):
        k = len(self.ts)
        self.H = np.zeros((6 * k, 6 * k))
        for i in range(k):
            ti = self.ts[i]
            self.H[6 * i:6 * i + 3, 6 * i:6 * i + 3] = np.array([
                [720 * ti ** 5, 360 * ti ** 4, 120 * ti ** 3],
                [360 * ti ** 4, 192 * ti ** 3, 72 * ti ** 2],
                [120 * ti ** 3, 72 * ti ** 2, 36 * ti],
            ])

    def cost_func(self, x):
        return x.T @ self.H @ x

    def linear_constraint_x(self, x):
        return self.A @ x - self.bs[:, 0]

    def linear_constraint_y(self, x):
        return self.A @ x - self.bs[:, 1]

    def linear_constraint_z(self, x):
        return self.A @ x - self.bs[:, 2]

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        if t > np.cumsum(self.Ti)[-1]:
            x = self.points[-1]
            flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                           'yaw': yaw, 'yaw_dot': yaw_dot}
            return flat_output

        i = 0
        for j in range(len(self.Ti)):
            if t < np.cumsum(self.Ti)[j]:
                i = j
                break

        t_sum = (t - self.start_true_ts[i]) / (self.start_true_ts[i + 1] - self.start_true_ts[i])

        ft = np.array([t_sum ** 5, t_sum ** 4, t_sum ** 3, t_sum ** 2, t_sum, 1])
        x = np.array([
            self.Cx[6 * i:6 * i + 6] @ ft,
            self.Cy[6 * i:6 * i + 6] @ ft,
            self.Cz[6 * i:6 * i + 6] @ ft
        ])

        ft = np.array([5 * t_sum ** 4, 4 * t_sum ** 3, 3 * t_sum ** 2, 2 * t_sum, 1, 0])
        x_dot = np.array([
            self.Cx[6 * i:6 * i + 6] @ ft,
            self.Cy[6 * i:6 * i + 6] @ ft,
            self.Cz[6 * i:6 * i + 6] @ ft,
        ])

        ft = np.array([20 * t_sum ** 3, 12 * t_sum ** 2, 6 * t_sum, 2, 0, 0])
        x_ddot = np.array([
            self.Cx[6 * i:6 * i + 6] @ ft,
            self.Cy[6 * i:6 * i + 6] @ ft,
            self.Cz[6 * i:6 * i + 6] @ ft,
        ])

        ft = np.array([60 * t_sum ** 2, 24 * t_sum, 6, 0, 0, 0])
        x_dddot = np.array([
            self.Cx[6 * i:6 * i + 6] @ ft,
            self.Cy[6 * i:6 * i + 6] @ ft,
            self.Cz[6 * i:6 * i + 6] @ ft,
        ])

        ft = np.array([120 * t_sum, 24, 0, 0, 0, 0])
        x_ddddot = np.array([
            self.Cx[6 * i:6 * i + 6] @ ft,
            self.Cy[6 * i:6 * i + 6] @ ft,
            self.Cz[6 * i:6 * i + 6] @ ft,
        ])

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output
def RDP_points(PointList, epsilon): # use Ramer-Douglas-Peucker algorithm to get the spase waypoints
    dmax = 0
    index = -1
    for i in range(1,len(PointList) - 1):
        d = WorldTraj.ver_dist(PointList[0],PointList[-1],PointList[i])
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:
        # Recursive call
        recResults1 = RDP_points(PointList[:index + 1], epsilon)
        recResults2 = RDP_points(PointList[index:], epsilon)

        # Build the result list
        ResultList = np.vstack((recResults1[:-1], recResults2))
    else:
        ResultList = np.vstack((PointList[0], PointList[-1]))

    return ResultList
