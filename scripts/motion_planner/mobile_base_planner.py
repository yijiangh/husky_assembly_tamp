import copy
import math
import random
import time

import kdtree
import numpy as np
import pybullet_planning as pp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from robot.robot_setup import INIT_ARM_JOINT_ANGLES, RobotSetup
from scipy.interpolate import CubicSpline


class RRTNode(object):
    def __init__(self, x, y, index=-1, father_id=-1):
        self.coords = (x, y)
        self.index = index
        self.father = father_id
        self.dist2father = 0

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return "Item({}, {}, {}, {})".format(self.coords[0], self.coords[1], self.index, self.father)

    def set_index(self, index):
        self.index = index

    def set_coords(self, x, y):
        self.coords = (x, y)

    def set_father(self, father_id):
        self.father = father_id

    def update_dist2father(self, father_node):
        dx = father_node.get_x() - self.get_x()
        dy = father_node.get_y() - self.get_y()
        self.dist2father = math.hypot(dx, dy)

    def get_x(self):
        return self.coords[0]

    def get_y(self):
        return self.coords[1]

    def get_index(self):
        return self.index

    def get_father(self):
        return self.father

    def get_dist2father(self):
        return self.dist2father

    def print_node(self):
        print("x:%f,y:%f,index:%d,father:%d" % (self.x, self.y, self.index, self.father))


class RRTStar(object):
    def __init__(
        self,
        expend_step,
        min_x,
        max_x,
        min_y,
        max_y,
        max_sample_time=120,
    ):
        self.expend_step = expend_step
        self.max_sample_time = max_sample_time
        self.sapmle_time = 0
        self.minx = min_x
        self.maxx = max_x
        self.miny = min_y
        self.maxy = max_y
        self.node_list = []

    def clear(self):
        self.node_list = []

    def plan(self, start_x, start_y, goal_x, goal_y, robot_setup: RobotSetup, collision_fn):
        self.clear()
        random.seed()
        start_node = RRTNode(start_x, start_y, 0)
        target_node = RRTNode(goal_x, goal_y)
        self.robot_setup = robot_setup
        # 生成RRT的kd树
        self.node_list.append(start_node)
        rrt_tree = kdtree.create(self.node_list)
        start_tick = time.time()
        cur_tick = time.time()
        is_reach_target = self.is_reach_target(start_node, target_node)
        id = 1
        # pp.wait_for_user("path plan break 1")
        while not is_reach_target and cur_tick - start_tick < self.max_sample_time:
            node_r = self.sample(goal_x, goal_y)
            node_near, dist = rrt_tree.search_nn(node_r)
            node_near: RRTNode = node_near.data
            dx = node_r.get_x() - node_near.get_x()
            dy = node_r.get_y() - node_near.get_y()
            theta = math.atan2(dy, dx)
            x_new = node_near.get_x() + self.expend_step * math.cos(theta)
            y_new = node_near.get_y() + self.expend_step * math.sin(theta)
            node_new = RRTNode(x_new, y_new)
            # pp.wait_for_user("path plan break 2")
            if self.is_collision_free_node(node_near, node_new, collision_fn):
                # pp.wait_for_user("path plan break 3")
                node_new.set_father(node_near.get_index())
                node_new.update_dist2father(node_near)
                node_new.set_index(id)
                r = 2 * node_new.get_dist2father()
                self.rrt_rewire(node_new, rrt_tree, r, collision_fn)
                rrt_tree.add(node_new)
                self.node_list.append(node_new)
                if not rrt_tree.is_balanced:
                    rrt_tree.rebalance()
                id += 1
                # 检测新节点与目标节点的距离是否满足终止条件
                is_reach_target = self.is_reach_target(node_new, target_node)
            cur_tick = time.time()
        if is_reach_target:
            target_node.set_father(node_new.get_index())
            target_node.update_dist2father(node_new)
            target_node.set_index(id)
            rrt_tree.add(target_node)
            self.node_list.append(target_node)
            if not rrt_tree.is_balanced:
                rrt_tree.rebalance()
            path_x, path_y = self.trace_back(target_node)
            self.path_smoothing(path_x, path_y, 5, collision_fn)
            path_x.reverse()
            path_y.reverse()
            return path_x, path_y
        else:
            return [], []

    def sample(self, goal_x, goal_y):
        prob = random.random()
        if prob > 0.70:
            x = random.randint(self.minx, self.maxx)
            y = random.randint(self.miny, self.maxy)
        else:
            x = goal_x
            y = goal_y
        node = RRTNode(x, y)
        return node

    def is_collision_free_node(self, node_start, node_end, collision_fn, steps=5, diagnosis=False):
        x_start = node_start.get_x()
        y_start = node_start.get_y()
        start_point = np.array([x_start, y_start])
        x_end = node_end.get_x()
        y_end = node_end.get_y()
        end_point = np.array([x_end, y_end])
        points = np.linspace(start_point, end_point, steps)
        for point in points:
            conf = point.tolist() + [0] + INIT_ARM_JOINT_ANGLES.tolist()
            # pp.wait_for_user(f"path plan break 2.1: {conf}")
            if collision_fn(conf, diagnosis):
                return False
        return True

    def is_collision_free_line(self, x_start, y_start, x_end, y_end, collision_fn, steps=5, diagnosis=False):
        start_point = np.array([x_start, y_start])
        end_point = np.array([x_end, y_end])
        points = np.linspace(start_point, end_point, steps)
        for point in points:
            conf = point.tolist() + [0] + INIT_ARM_JOINT_ANGLES.tolist()
            if collision_fn(conf, diagnosis):
                return False
        return True

    def rrt_rewire(self, node_new, rrt_tree, r, collision_fn):
        dist_cur = self.get_total_cost(node_new)
        r_sq = math.pow(r, 2)
        node_potential = []
        node_potential = rrt_tree.search_nn_dist(node_new, r_sq)
        dist_list = []
        for item in node_potential:
            dist = self.get_total_cost(item)
            dx = node_new.get_x() - item.get_x()
            dy = node_new.get_y() - item.get_y()
            dist += math.hypot(dx, dy)
            dist_list.append(dist)
        while len(node_potential):
            dist_min = min(dist_list)
            dist_min_id = dist_list.index(dist_min)
            if dist_min < dist_cur:
                if self.is_collision_free_node(node_potential[dist_min_id], node_new, collision_fn):
                    node_new.set_father(node_potential[dist_min_id].get_index())
                    node_new.update_dist2father(node_potential[dist_min_id])
                    return
                else:
                    dist_list.pop(dist_min_id)
                    node_potential.pop(dist_min_id)
            else:
                return

    def get_total_cost(self, node):
        cur_node = node
        total_dist = 0
        while cur_node.get_father() != -1:
            total_dist += cur_node.get_dist2father()
            cur_node = self.node_list[cur_node.get_father()]
        return total_dist

    def is_reach_target(self, cur_node, target_node):
        dx = target_node.get_x() - cur_node.get_x()
        dy = target_node.get_y() - cur_node.get_y()
        dist = math.hypot(dx, dy)
        if dist < self.expend_step:
            return True
        else:
            return False

    def trace_back(self, target_node):
        cur_node = target_node
        x_list = []
        y_list = []
        while cur_node.get_father() != -1:
            x_list.append(cur_node.get_x())
            y_list.append(cur_node.get_y())
            cur_node = self.node_list[cur_node.get_father()]
        x_list.append(cur_node.get_x())
        y_list.append(cur_node.get_y())
        return x_list, y_list

    def get_obstacle_tree(self):
        return copy.deepcopy(self.obstacle_tree)

    def path_smoothing(self, path_x, path_y, iter, collision_fn):
        if len(path_x) < 5:
            return
        for _ in range(0, iter):
            for id in range(0, len(path_x) - 4):
                x__2 = path_x[id]
                y__2 = path_y[id]
                x__1 = path_x[id + 1]
                y__1 = path_y[id + 1]
                x_0 = path_x[id + 2]
                y_0 = path_y[id + 2]
                x_1 = path_x[id + 3]
                y_1 = path_y[id + 3]
                x_2 = path_x[id + 4]
                y_2 = path_y[id + 4]
                dx = -1 / 6 * (x__2 - 4 * x__1 + 6 * x_0 - 4 * x_1 + x_2)
                dy = -1 / 6 * (y__2 - 4 * y__1 + 6 * y_0 - 4 * y_1 + y_2)
                if self.is_collision_free_line(
                    x__1, y__1, x_0 + dx, y_0 + dy, collision_fn
                ) and self.is_collision_free_line(x_0 + dx, y_0 + dy, x_1, y_1, collision_fn):
                    path_x[id + 2] = x_0 + dx
                    path_y[id + 2] = y_0 + dy

    def update_obtree(self, ob_x_list, ob_y_list):
        ob_node_list = [RRTNode(x, y) for (x, y) in zip(ob_x_list, ob_y_list)]
        try:
            type(self.obstacle_tree)
        except:
            self.obstacle_tree = kdtree.create(ob_node_list)
        else:
            del self.obstacle_tree
            self.obstacle_tree = kdtree.create(ob_node_list)
        if not self.obstacle_tree.is_balanced:
            self.obstacle_tree.rebalance()

    # def plot(self, x_list, y_list):
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(self.minx, self.maxx)
    #     ax.set_ylim(self.miny, self.maxy)
    #     ax.set_aspect("equal")

    #     for obs_x, obs_y in zip(self.ob_x_list, ob_y_list):
    #         # rect = Rectangle((obs_x, obs_y), self.avoid_dist, self.avoid_dist, color='gray')
    #         rect = Circle((obs_x, obs_y), self.avoid_dist, color="gray")
    #         ax.add_patch(rect)

    #     # tree_start_x, tree_start_y = zip(*self.tree_start)
    #     # tree_goal_x, tree_goal_y = zip(*self.tree_goal)

    #     ax.plot(x_list[0], y_list[0], "g.")
    #     ax.plot(x_list[-1], y_list[-1], "r.")

    #     ax.plot(x_list, y_list, "b-", lw=2)

    #     ax.plot(x_list[0], y_list[0], "go", markersize=10)
    #     ax.plot(x_list[-1], y_list[-1], "ro", markersize=10)

    #     plt.show()


def fill_yaw_angle(goal_yaw, x_list, y_list):
    n_points = len(x_list)

    yaw_list = np.zeros(n_points)

    for i in range(0, n_points - 1):
        dx = x_list[i + 1] - x_list[i]
        dy = y_list[i + 1] - y_list[i]
        yaw_list[i] = np.arctan2(dy, dx)

    yaw_list[-1] = goal_yaw

    return yaw_list


def extract_obstacles_from_bars(bars, resolution=10):
    obs_x_list = []
    obs_y_list = []
    for bar in bars:
        bar_start = np.array(bar[0])
        bar_end = np.array(bar[1])
        bar_points = np.linspace(bar_start, bar_end, resolution)
        for bar_point in bar_points:
            obs_x_list.append(bar_point[0])
            obs_y_list.append(bar_point[1])
    return obs_x_list, obs_y_list


if __name__ == "__main__":
    x_range = (-3, 3)
    y_range = (-3, 3)
    rrt_star = RRTStar(0.05, *x_range, *y_range)

    edge_1_y = list(np.arange(-3, 3, 0.25))
    edge_2_x = list(np.arange(-3, 3, 0.25))
    edge_3_y = list(np.arange(3, -3, -0.25))
    edge_4_x = list(np.arange(3, -3, -0.25))

    ob_x_list = [-3] * len(edge_1_y) + edge_2_x + [3] * len(edge_3_y) + edge_4_x
    ob_y_list = edge_1_y + [3] * len(edge_2_x) + edge_3_y + [-3] * len(edge_4_x)

    edge_5_y = list(np.arange(0, 3, 0.25))
    ob_x_list = ob_x_list + [0] * len(edge_5_y)
    ob_y_list = ob_y_list + edge_5_y

    # ob_x_list = [2, 3, 4, 5, 6, 7, 8]
    # ob_y_list = [2, 3, 4, 5, 6, 7, 8]
    x_list = []
    y_list = []
    # start_pose = (-0.84130859375, 0.7238764762878418)
    # goal_pose = (12.9344029426574707, 12.9344029426574707)
    start_pose = (1.5, 1.5)
    goal_pose = (-1.5, 1.5)
    x_list, y_list = rrt_star.plan(copy.deepcopy(ob_x_list), copy.deepcopy(ob_y_list), *start_pose, *goal_pose)
    # print(x_list)
    # print(y_list)
    # rrt_star.plot(x_list, y_list)
