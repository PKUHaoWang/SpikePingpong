import numpy as np
import math
from collections import deque

class BallVelocityEstimator:
    def __init__(self, history_size=15, mixing_constants=(0.5, 0.5, 0.5)): #mixing_constants=(0.15, 0.15, 0.25)
        self.history = deque(maxlen=history_size)
        self.mixing = mixing_constants

        self.rho = 1.225  # 空气密度(kg/m^3)，海平面标准值
        self.Cd = 0.47    # 阻力系数，根据乒乓球调整
        self.ball_mass = 0.0027  # 乒乓球质量(kg)
        self.ball_radius = 0.02  # 乒乓球半径(m)
        
    def add_position(self, time, position):
        if len(self.history) > 0:
            last_time = self.history[-1][0]
            if time - last_time > 1.0:
                self.history.clear()
        
        self.history.append((time, position))
        
    def estimate_velocity(self):
        if len(self.history) < 2:
            return None, None
            
        times = []
        positions = [[], [], []]
        
        for t, pos in self.history:
            times.append(t)
            for i in range(3):
                positions[i].append(pos[i])
                
        times = np.array(times)
        
        filtered_pos = []
        filtered_vel = []
        
        directions = ['x', 'y', 'z']
        for i, (pos_array, direction, mix) in enumerate(zip(positions, directions, self.mixing)):
            pos_filtered, vel_filtered = self._ema(pos_array, times, direction, mix)
            filtered_pos.append(pos_filtered[-1])
            filtered_vel.append(vel_filtered[-1])
            
        return filtered_pos, filtered_vel
        
    def _ema(self, pos, time, direction, mix_constant):
        acc = 0
        if direction == 'x' or direction == 'y':
            acc = 0
        elif direction == 'z':
            acc = -9.81
            
        pos_filtered = np.array(pos[0:2])
        vel_filtered = np.array([(pos[1] - pos[0]) / (time[1] - time[0])])
        
        count_z = 0
        
        for i in range(2, len(time)):
            dt = time[i] - time[i-1]
            # print(dt)
            cur_vel = (pos[i] - pos[i-1]) / dt
            
            if direction == 'z' and cur_vel > 0 and count_z < 3:
                vel_filtered = np.append(vel_filtered, cur_vel)
                pos_filtered = np.append(pos_filtered, pos[i])
                count_z += 1
                continue
                
            pred_vel = vel_filtered[-1] + acc * dt
            est_vel = (1 - mix_constant) * pred_vel + mix_constant * cur_vel
            vel_filtered = np.append(vel_filtered, est_vel)
            
            pred_pos = pos_filtered[-1] + vel_filtered[-2] * dt + 0.5 * acc * dt**2
            est_pos = (1 - mix_constant) * pred_pos + mix_constant * pos[i]
            pos_filtered = np.append(pos_filtered, est_pos)
            
        return pos_filtered, vel_filtered


class BallTrajectoryPredictor:
    def __init__(self, y_end=-1.4):
        # Table parameters
        self.table_height = 0.76
        self.table_length = 2.74
        self.table_width = 1.525
        self.e = 0.87                 # coefficient of restitution
        self.ball_radius = 0.02
        self.g = -9.81

        # Robot parameters
        self.robot_pos_y = -(1.37 + 0.45)
        self.y_end = y_end    # target y position
        self.z_end = 1.0      # target z position
        self.ws_radius = 1.0 # 0.58  # robot workspace radius
        self.x_limit_squared = self.ws_radius**2 - (abs(self.robot_pos_y) - abs(self.y_end))**2

        # Prediction region
        self.y_net = 0
        self.y_begin = -0.65 # 0.23   # start prediction when ball passes this y
        self.y_predict = -0.8 # -0.57 # stop prediction when ball passes this y

    def predict_trajectory(self, current_time, position, velocity):
        """
        Predict the final position and velocity of the ball.
        
        Parameters:
            current_time: Current timestamp
            position: [x, y, z] position after filtering
            velocity: [vx, vy, vz] velocity after filtering
            
        Returns:
            Dictionary with prediction results, including:
            - hittable: bool, whether the ball can be hit
            - end_position: [x, y, z] predicted end position
            - end_velocity: [vx, vy, vz] predicted end velocity
            - end_time: predicted time when ball reaches end position
        """
        x, y, z = position
        vx, vy, vz = velocity
        y_end = self.y_end
        z_end = self.z_end
        
        result = {
            'hittable': False,
            'end_position': None,
            'end_velocity': None,
            'end_time': None,
            'message': ''
        }

        # Check if ball is moving toward robot
        if vy >= 0:
            result['message'] = 'Ball not moving toward robot'
            return result

        # Check if ball is on the table
        if (z < self.table_height + self.ball_radius or
                abs(x) > self.table_width/2 or
                abs(y) > self.table_length/2):
            result['message'] = 'Ball not on table'
            return result

        # Check if in prediction region
        if y > self.y_begin:
            result['message'] = 'Too early to predict'
            return result
            
        if y < self.y_predict:
            result['message'] = 'Ball past prediction line'
            return result

        # Time for ball to reach y_end
        t = (y_end - y) / vy
        if t < 0:
            result['message'] = 'Ball moving in wrong direction or past end'
            return result

        # Check if final x position is within workspace
        x_end = x + vx * t
        if x_end**2 > self.x_limit_squared:
            result['message'] = f'Final x-position outside of workspace: {x_end}'
            return result

        # Calculate final z position without rebound
        z_temp = z + vz * t + 0.5 * self.g * t**2
        
        # if z_temp > self.table_height + self.ball_radius:
        #     # Ball reaches final y position without rebound
        #     if y > -self.table_length/2:
        #         end_position = [x_end, y_end, z_temp]
        #         end_velocity = [vx, vy, vz + t * self.g]
        #         end_time = current_time + t
                
        #         result['hittable'] = True
        #         result['end_position'] = end_position
        #         result['end_velocity'] = end_velocity
        #         result['end_time'] = end_time
        #         result['message'] = 'Ball did not hit the table, using estimated state'
        #         return result
        #     else:
        #         result['message'] = f'Ball will not hit the table: {z_temp}'
        #         return result

        # Otherwise, ball rebounds at least once
        
        # Find time for first rebound
        t_rb1 = (-vz - math.sqrt(vz**2 - 2 * self.g * (z - self.table_height - self.ball_radius))) / self.g
        
        # Find z velocity just before and after rebound
        vz_in = -math.sqrt(-2 * self.g * (z - self.table_height - self.ball_radius) + vz**2)
        vz_out = -self.e * vz_in

        # Time for second rebound
        t_rb2 = -2 * vz_out / self.g
        
        # Time remaining after first rebound
        t_rem = t - t_rb1
        
        # if t_rb2 < t_rem:
        #     # Second rebound before reaching y_end
        #     # Solve quadratic equation for when ball reaches z_end
        #     try:
        #         discrim = math.sqrt(vz_out**2 + 2 * self.g * (z_end - self.table_height - self.ball_radius))
        #         time1, time2 = (-vz_out + discrim) / self.g, (-vz_out - discrim) / self.g
                
        #         t_hit = None
        #         choose1, choose2 = True, True
                
        #         if time1 < 0 or time1 > t_rem or abs(y + (t_rb1 + time1) * vy) > self.ws_radius:
        #             choose1 = False
        #         if time2 < 0 or time2 > t_rem or abs(y + (t_rb1 + time2) * vy) > self.ws_radius:
        #             choose2 = False
                    
        #         if not choose1 and not choose2:
        #             result['message'] = 'No valid solutions for hitting time'
        #             return result
        #         elif choose1 and choose2:
        #             t_hit = max(time1, time2)
        #         elif choose1:
        #             t_hit = time1
        #         elif choose2:
        #             t_hit = time2
                
        #         t_end = t_rb1 + t_hit
        #         x_end = x + vx * t_end
        #         y_end = y + vy * t_end
        #         vz_end = vz_out + self.g * t_hit
                
        #         result['hittable'] = True
        #         result['end_position'] = [x_end, y_end, z_end]
        #         result['end_velocity'] = [vx, vy, vz_end]
        #         result['end_time'] = current_time + t_end
        #         result['message'] = 'Multiple rebounds, found valid hitting time'
        #         return result
                
        #     except Exception as e:
        #         result['message'] = f'Error calculating hitting time: {str(e)}'
        #         return result
        # else:
        #     # One rebound only
        #     z_end_actual = self.table_height + self.ball_radius + vz_out * t_rem + 0.5 * self.g * t_rem**2
        #     vz_end = vz_out + self.g * t_rem
            
        #     result['hittable'] = True
        #     result['end_position'] = [x_end, y_end, z_end_actual]
        #     result['end_velocity'] = [vx, vy, vz_end]
        #     result['end_time'] = current_time + t
        #     result['message'] = 'Ball rebounds once and is hittable'
        #     return result
        

        # One rebound only
        z_end_actual = self.table_height + self.ball_radius + vz_out * t_rem + 0.5 * self.g * t_rem**2
        vz_end = vz_out + self.g * t_rem
        
        result['hittable'] = True
        result['end_position'] = [x_end, y_end, z_end_actual]
        result['end_velocity'] = [vx, vy, vz_end]
        result['end_time'] = current_time + t
        result['message'] = 'Ball rebounds once and is hittable'
        return result


# 使用示例
def main():
    # 创建速度估计器和轨迹预测器
    velocity_estimator = BallVelocityEstimator()
    trajectory_predictor = BallTrajectoryPredictor()
    
    # 假设这是您获取数据的循环
    while True:
        # 获取当前时间和位置 (这部分需要替换为您的数据源)
        current_time = get_current_time()  # 获取当前时间戳
        position = get_ball_position()     # 获取[x, y, z]位置
        
        # 添加位置数据到速度估计器
        velocity_estimator.add_position(current_time, position)
        
        # 估计速度
        filtered_pos, filtered_vel = velocity_estimator.estimate_velocity()
        
        if filtered_pos is not None:
            # 进行轨迹预测
            prediction = trajectory_predictor.predict_trajectory(
                current_time, filtered_pos, filtered_vel)
            
            # 输出预测结果
            if prediction['hittable']:
                print("可击打！预测位置:", prediction['end_position'])
                print("预测速度:", prediction['end_velocity'])
                print("预测时间:", prediction['end_time'])
            else:
                print("无法击打:", prediction['message'])