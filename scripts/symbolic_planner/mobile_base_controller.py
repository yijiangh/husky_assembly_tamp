from typing import List, Tuple

import numpy as np
import pybullet_planning as pp
from utils import angle_distance


class State:
    def __init__(self, x_, y_, yaw_):
        if not (x_ is None or y_ is None or yaw_ is None):
            self.x = x_
            self.y = y_
            self.yaw = yaw_
        else:
            self.x = 0
            self.y = 0
            self.yaw = 0

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.yaw)


class Controller(object):
    def __init__(self, start: State, path_points: List[State], dt: float = 0.1, **kwargs) -> None:
        self.start = start
        self.path_points = path_points
        self.dt = dt

    def run(self) -> Tuple[List[float], List[float], List[float]]:
        raise NotImplementedError()

    def step(self, current: State, target: State) -> Tuple[float, float]:
        raise NotImplementedError()

    def take_action(self, current: State, v: float, w: float) -> Tuple[float, float, float]:
        x = current.x
        y = current.y
        yaw = current.yaw

        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += w * self.dt

        return x, y, yaw


class PID(Controller):
    def __init__(self, start: State, path_points: List[State], dt: float = 0.1, **kwargs) -> None:
        super().__init__(start, path_points, dt, **kwargs)
        self.kp = kwargs.get("kp", 10.0)
        self.ki = kwargs.get("ki", 0.01)
        self.kd = kwargs.get("kd", 0.01)
        self.switch_distance = kwargs.get("switch_distance", 0.1)
        self.v_max = kwargs.get("max_velocity", 0.5)

        self.e = 0  # Cummulative error
        self.old_e = 0  # Previous error

        self.is_start = False
        self.is_last = False

    def run(self) -> Tuple[List[float], List[float], List[float]]:
        current = self.start
        x = []
        y = []
        yaw = []
        num_targets = len(self.path_points)
        traj = 0
        self.is_start = True
        for target in self.path_points:
            traj += 1
            if traj == num_targets:
                self.is_last = True
            else:
                self.is_last = False
            x_, y_, theta_, current = self.run_single(current, target)
            x.extend(x_)
            y.extend(y_)
            yaw.extend(theta_)
            self.is_start = False
        return x, y, yaw

    def step(self, current: State, target: State) -> Tuple[float, float]:
        # Difference in x and y
        d_x = target.x - current.x
        d_y = target.y - current.y

        # Angle from robot to goal
        g_theta = np.arctan2(d_y, d_x)

        # Error between the goal angle and robot angle
        alpha = g_theta - current.yaw
        e = np.arctan2(np.sin(alpha), np.cos(alpha))

        e_P = e
        e_I = self.e + e
        e_D = e - self.old_e

        # This PID controller only calculates the angular
        # velocity with constant speed of v
        # The value of v can be specified by giving in parameter or
        # using the pre-defined value defined above.
        w = self.kp * e_P + self.ki * e_I + self.kd * e_D

        w = np.arctan2(np.sin(w), np.cos(w))

        self.e = self.e + e
        self.old_e = e
        v = self.v_max

        if self.is_start and np.abs(e) >= 0.01:
            w = 1.0 * e
            v = 0  # 仅调整朝向，不再前进

        # 如果距离已经接近目标点，则仅调整朝向
        if np.hypot(d_x, d_y) < self.switch_distance and self.is_last:
            angle_error = target.yaw - current.yaw
            e = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            w = 1.0 * e
            v = 0  # 仅调整朝向，不再前进

        return v, w

    def run_single(self, current: State, target: State) -> Tuple[List[float], List[float], List[float], State]:
        x = [current.x]
        y = [current.y]
        yaw = [current.yaw]
        while not self.is_arrived(current, target):
            v, w = self.step(current, target)
            next_x, next_y, next_yaw = self.take_action(current, v, w)
            x.append(next_x)
            y.append(next_y)
            yaw.append(next_yaw)
            current = State(next_x, next_y, next_yaw)
        return x, y, yaw, current

    def is_arrived(self, current: State, target: State):
        current_state = np.array([current.x, current.y])
        goal_state = np.array([target.x, target.y])
        difference = current_state - goal_state

        distance_err = difference @ difference.T
        angle_err = np.abs(self.format_angle(target.yaw - current.yaw))
        if distance_err < self.switch_distance and (not self.is_last or angle_err < 0.01):
            return True
        else:
            return False

    def format_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))


class Stanley(Controller):
    def __init__(self, start: State, path_points: List[State], dt: float = 0.1, **kwargs) -> None:
        super().__init__(start, path_points, dt, **kwargs)
        self.max_steps = kwargs.get("max_steps", 2000)
        # self.switch_distance = kwargs.get("switch_distance", 0.1)
        self.v_max = kwargs.get("max_velocity", 1)
        self.w_max = kwargs.get("max_angle_velocity", np.pi / 2)
        self.stanley_gain = kwargs.get("stanley_gain", 0.75)
        self.position_tolerance = kwargs.get("position_tolerance", 0.1)
        self.yaw_tolerance = kwargs.get("yaw_tolerance", 0.05)

        self.cur_step = 0

    def run(self) -> Tuple[List[float], List[float], List[float]]:
        current = self.start
        # final = self.path_points[-1]
        robot_positions = []
        target_index = 0
        while True:
            # print("==========")
            linear_speed, angular_speed, target_index = self.step(current, target_index)
            # print("current ", current.x, current.y, current.yaw)
            # print("linear ", linear_speed, "angular ", angular_speed, "index ", target_index)
            # pp.wait_for_user("Stanley controller break")

            # 更新机器人位置
            next_x, next_y, next_yaw = self.take_action(current, linear_speed, angular_speed)

            self.cur_step += 1

            robot_positions.append((next_x, next_y, next_yaw))

            current = State(next_x, next_y, next_yaw)

            if self.cur_step >= self.max_steps:
                break
            if target_index >= len(self.path_points):
                break

        path_x = [pos[0] for pos in robot_positions]
        path_y = [pos[1] for pos in robot_positions]
        path_yaw = [pos[2] for pos in robot_positions]
        return path_x, path_y, path_yaw

    def step(self, current: State, target_index: int) -> Tuple[float, float, int]:
        if target_index >= len(self.path_points):
            target_index = len(self.path_points) - 1

        target = self.path_points[target_index]

        target_x, target_y, target_yaw = target.x, target.y, target.yaw

        dx = target_x - current.x
        dy = target_y - current.y
        heading_error = angle_distance(np.arctan2(dy, dx), current.yaw)

        angular_speed = heading_error
        if angular_speed > self.w_max:
            angular_speed = self.w_max
        elif angular_speed < -self.w_max:
            angular_speed = -self.w_max

        distance_to_target = np.sqrt(dx**2 + dy**2)
        # print("distance_to_target ", distance_to_target)
        yaw_error = angle_distance(target_yaw, current.yaw)
        # print("yaw_error ", yaw_error)
        if distance_to_target <= self.position_tolerance and abs(yaw_error) > self.yaw_tolerance:
            linear_speed = 0
            angular_speed = yaw_error
            if angular_speed > self.w_max:
                angular_speed = self.w_max
            elif angular_speed < -self.w_max:
                angular_speed = -self.w_max
        elif distance_to_target <= self.position_tolerance and abs(yaw_error) <= self.yaw_tolerance:
            linear_speed = 0
            angular_speed = 0
            target_index += 1
        else:
            linear_speed = self.v_max / (1 + self.stanley_gain * abs(angular_speed))
        
        return linear_speed, angular_speed, target_index

        # # 计算横向误差
        # dx = target_x - current.x
        # dy = target_y - current.y
        # cross_track_error = -np.sin(target_yaw) * dx + np.cos(target_yaw) * dy

        # # 计算转向角
        # heading_error = angle_distance(target_yaw, current.yaw)
        # # heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize the angle to [-pi, pi]

        # # Stanley 控制律
        # control_steering = heading_error + np.arctan2(self.stanley_gain * cross_track_error, self.v_max)

        # # 限制角速度
        # angular_speed = control_steering
        # if angular_speed > self.w_max:
        #     angular_speed = self.w_max
        # elif angular_speed < -self.w_max:
        #     angular_speed = -self.w_max

        # # 线速度与角速度相关联，以避免在初始yaw角相差过大的情况下机器人以最大线速度转圈
        # linear_speed = self.v_max / (1 + self.stanley_gain * abs(angular_speed))

        # # 检查是否需要切换到下一个目标点
        # distance_to_target = np.sqrt(dx**2 + dy**2)
        # if distance_to_target < self.switch_distance:
        #     target_index += 1

        # return linear_speed, angular_speed, target_index
