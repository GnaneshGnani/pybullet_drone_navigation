import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BulletNavigationEnv(gym.Env):
    def __init__(self, waypoints, use_camera = False, use_depth = False, use_lidar = False, 
                 use_obstacles = False, waypoint_threshold = 1.0, waypoint_bonus = 100.0,
                 crash_penalty = -100.0, timeout_penalty = -10.0, per_step_penalty = -0.1,
                 max_dist_from_target = 10.0, action_limits = None, gui = False, show_waypoints = False):
        
        # Store a copy of the initial waypoints to detect the task type
        self.waypoints = waypoints
        
        self.use_camera = use_camera
        self.use_depth = use_depth
        self.use_lidar = use_lidar
        self.use_obstacles = use_obstacles

        self.max_dist_from_target = max_dist_from_target
        self.show_waypoints = show_waypoints
        
        # Reward Config
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_bonus = waypoint_bonus
        self.crash_penalty = crash_penalty
        self.timeout_penalty = timeout_penalty
        self.per_step_penalty = per_step_penalty
        
        # State Augmentation: Previous Action
        self.prev_action = np.zeros(4)
        
        # Visualization IDs
        self.marker_ids = [] 
        self.debug_line_ids = []

        # PyBullet Drone Init
        self.env = CtrlAviary(
            drone_model = DroneModel.CF2X,
            num_drones = 1,
            neighbourhood_radius = 10,
            physics = Physics.PYB,
            gui = gui,
            pyb_freq = 240,
            ctrl_freq = 48 
        )

        self.ctrl = DSLPIDControl(drone_model = DroneModel.CF2X)
        
        # Action Space
        # [Target Vx, Target Vy, Target Vz, Target Yaw Rate]
        self.alpha = 0.75  # Smoothing factor (0.1 = very smooth/sluggish, 0.9 = responsive/jerky)
        self.smooth_action = np.zeros(4)
        self.action_limits = np.array(action_limits) if action_limits is not None else np.array([1.0, 1.0, 1.0, 1.0])

        self.action_space = gym.spaces.Box(low = -1, high = 1, shape = (4,), dtype = np.float32)
        
        self.state_dim = 18
        self.lidar_rays = 360

        obs_spaces = {
            "state": gym.spaces.Box(low = -np.inf, high = np.inf, shape = (self.state_dim,), dtype = np.float32)
        }
        
        if self.use_camera or self.use_depth:
            c = (3 if self.use_camera else 0) + (1 if self.use_depth else 0)
            obs_spaces["image"] = gym.spaces.Box(low = 0, high = 1, shape = (c, 64, 64), dtype = np.float32)
        
        if self.use_lidar:
            obs_spaces["lidar"] = gym.spaces.Box(low = 0, high = 10, shape = (self.lidar_rays,), dtype = np.float32)
        
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, seed = None, options = None):
        self.env.reset()
        
        if len(self.waypoints) == 0:
            raise ValueError("No waypoints provided.")
        
        self._draw_waypoints()
        
        self.current_wp_idx = 0
        self.target_pos = self.waypoints[self.current_wp_idx]
        self.prev_dist = np.linalg.norm(self._get_drone_pos() - self.target_pos)
        self.prev_action = np.zeros(4) 
        self.smooth_action = np.zeros(4)
        
        if self.use_obstacles:
            self._spawn_obstacles()

        self.ctrl.reset()

        return self._get_observation(), {}

    def _draw_waypoints(self):
        if not self.show_waypoints:
            return
        
        if p.getConnectionInfo(physicsClientId = self.env.CLIENT)["connectionMethod"] == p.GUI:
            self.marker_ids = []
            self.debug_line_ids = []

            for i, wp in enumerate(self.waypoints):
                # Only draw the first waypoint for hover tasks, or all for pathing
                if len(self.waypoints) > 1 and np.allclose(self.waypoints[0], self.waypoints[-1]) and i > 0:
                    break 

                visual_shape_id = p.createVisualShape(
                    shapeType = p.GEOM_SPHERE,
                    radius = 0.1,
                    rgbaColor = [0, 1, 0, 1],
                    physicsClientId = self.env.CLIENT
                )
                
                body_id = p.createMultiBody(
                    baseMass = 0,
                    baseCollisionShapeIndex = -1, # No Collision Shape
                    baseVisualShapeIndex = visual_shape_id,
                    basePosition = wp,
                    physicsClientId = self.env.CLIENT
                )
                self.marker_ids.append(body_id)

                if i < len(self.waypoints) - 1:
                    next_wp = self.waypoints[i+1]
                    line_id = p.addUserDebugLine(
                        lineFromXYZ = wp,
                        lineToXYZ = next_wp,
                        lineColorRGB = [1, 0, 0],
                        lineWidth = 3.0,
                        physicsClientId = self.env.CLIENT
                    )
                    self.debug_line_ids.append(line_id)

    def _spawn_obstacles(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for _ in range(6):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            z = np.random.uniform(3.0, 7.0)
            if np.linalg.norm([x, y]) > 1.0:
                p.loadURDF("cube.urdf", [x, y, z], globalScaling = 0.6, physicsClientId = self.env.CLIENT)

    def step(self, action):  
        self.smooth_action = self.alpha * action + (1 - self.alpha) * self.smooth_action    
        scaled_action = self.smooth_action * self.action_limits

        pos, quat = p.getBasePositionAndOrientation(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        lin_vel, ang_vel = p.getBaseVelocity(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)

        current_pos = np.array(pos)
        lin_vel_np = np.array(lin_vel)

        r = R.from_quat(quat)

        action_body = np.array(scaled_action[:3])
        target_v_world = r.apply(action_body)

        target_yaw_rate = scaled_action[3] 
        
        # Compute RPMs using DSLPIDControl
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep = 1.0 / self.env.CTRL_FREQ,

            cur_pos = current_pos,
            cur_quat = np.array(quat),
            cur_vel = lin_vel_np,
            cur_ang_vel = np.array(ang_vel),

            target_pos = current_pos, # Keep current pos target to force velocity tracking
            target_vel = target_v_world,

            target_rpy = np.array([0, 0, 0]),
            target_rpy_rates = np.array([0, 0, target_yaw_rate])
        )
        
        # Step the environment with calculated RPMs
        self.env.step(rpm.reshape(1, 4))

        # OBSERVE & REFRESH STATE (Time t+1)
        # CRITICAL: We must re-read position to calculate reward for the NEW state
        pos, quat = p.getBasePositionAndOrientation(self.env.DRONE_IDS[0], physicsClientId=self.env.CLIENT)
        lin_vel, ang_vel = p.getBaseVelocity(self.env.DRONE_IDS[0], physicsClientId=self.env.CLIENT)
        
        current_pos = np.array(pos)
        lin_vel_np = np.array(lin_vel)

        obs = self._get_observation()

        reward = 0
        terminated = False
        truncated = False
        
        # Per-Step Penalty (REVISED: Milder penalty for smoother flight preference)
        reward += (self.per_step_penalty)
        
        dist = np.linalg.norm(current_pos - self.target_pos)

        progress = self.prev_dist - dist
        reward += 30.0 * progress

        # Acceleration/Jerk Penalty: Penalize changing motor commands too quickly
        diff_action = action - self.prev_action
        reward -= 1.0 * np.linalg.norm(diff_action) ** 2

        # High Velocity Penalty
        reward -= 0.1 * np.linalg.norm(lin_vel) ** 2
        reward -= 0.1 * np.linalg.norm(ang_vel) ** 2

        if dist < self.waypoint_threshold:
            reward += self.waypoint_bonus
            self.current_wp_idx += 1

            if self.current_wp_idx >= len(self.waypoints):
                print("--- Course Complete! ---")
                reward += 100.0
                terminated = True

            else:
                self.target_pos = self.waypoints[self.current_wp_idx]
                print(f"--- Reached Waypoint {self.current_wp_idx} ---")
                self.prev_dist = np.linalg.norm(current_pos - self.target_pos)

        # Penalize the CHANGE in action (Jerk). 
        # If the drone twitches (changes action from 0.1 to 0.9), this penalty is high.
        # diff_action = action - self.prev_action
        # smooth_reward = -0.05 * np.linalg.norm(diff_action) ** 2
        # reward += smooth_reward

        # We want the drone to be still (hovering) when it arrives.
        # We penalize high speeds, but we scale it by proximity.
        # (It's okay to move fast if you are far away, but you must stop when close).
        # This prevents the drone from just flying continuously through the point.
        # high_speed_reward = -0.1 * (np.linalg.norm(lin_vel) ** 2)
        # high_speed_reward += -0.1 * (np.linalg.norm(ang_vel) ** 2)
        # reward += high_speed_reward

        # If you care about facing a specific direction:
        # Convert quat to Euler to get Yaw
        # euler = p.getEulerFromQuaternion(quat)
        # current_yaw = euler[2]
        
        # delta_x = self.target_pos[0] - current_pos[0]
        # delta_y = self.target_pos[1] - current_pos[1]

        # Compute the angle (yaw) required to face that point
        # atan2 handles the quadrants correctly (-pi to +pi)
        # if np.linalg.norm([delta_x, delta_y]) > 0.1:
        #     desired_yaw = np.arctan2(delta_y, delta_x)
        # else:
        #     # If we are effectively AT the target, keep the current heading
        #     # so the drone doesn't spin wildly trying to face itself.
        #     desired_yaw = current_yaw
        # # Calculate shortest angular distance (-pi to pi)
        # yaw_error = (current_yaw - desired_yaw + np.pi) % (2 * np.pi) - np.pi
        # orientation_reward = -0.1 * abs(yaw_error)
        # reward += orientation_reward

        # Discourage maximizing motors if not necessary (prevents saturation)
        # energy_reward = -0.05 * np.linalg.norm(action) ** 2
        # reward += energy_reward

        contact_points = p.getContactPoints(bodyA=self.env.DRONE_IDS[0], physicsClientId=self.env.CLIENT)
        if len(contact_points) > 0:
            print("Crashed!")
            reward += self.crash_penalty
            terminated = True

        if dist > self.max_dist_from_target:
            print("Timeout!")
            reward += self.timeout_penalty
            terminated = True
            
        if abs(current_pos[0]) > 20 or abs(current_pos[1]) > 20 or current_pos[2] > 10:
            print("Out of Boundary!")
            terminated = True

        # Store the current action for the next time step's observation
        self.prev_action = np.array(action)
        self.prev_dist = dist
        
        info = {
            "waypoints_reached": self.current_wp_idx,
            "total_waypoints": len(self.waypoints),
            "dist_to_current_target": dist
        }

        return obs, reward, terminated, truncated, info

    def _get_drone_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        return np.array(pos)

    def _get_observation(self):
        pos, quat = p.getBasePositionAndOrientation(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        lin_vel, ang_vel = p.getBaseVelocity(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        
        pos = np.array(pos)
        quat = np.array(quat)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)

        r = R.from_quat(quat)
        r_inv = r.inv()
        rot_mat = r.as_matrix()
        
        target_vec_world = self.target_pos - pos
        target_vec_body = r_inv.apply(target_vec_world)

        dist = np.linalg.norm(target_vec_body)
        dist_feat = np.log(dist + 1.0)

        if dist > 1e-5:
            target_direction = target_vec_body / dist
        else:
            target_direction = np.zeros(3)

        lin_vel_body = r_inv.apply(lin_vel)

        # Use clip range of 2.0 (since max speed is ~0.75)
        # This spreads the data out so the network can "see" speed better
        lin_vel_scaled = np.clip(lin_vel_body, -2.0, 2.0) / 2.0

        # Use clip range of 3.0 (since max yaw is ~1.0)
        # 10.0 was way too high; the drone would be spinning out of control at that speed
        ang_vel_scaled = np.clip(ang_vel, -3.0, 3.0) / 3.0

        # target vector, orientation, velocities, and previous action
        state_vec = np.concatenate([
            target_direction,    # 3 dims
            [dist_feat],         # 1 dim
            quat,                # 4 dims
            lin_vel_scaled,      # 3 dims
            ang_vel_scaled,      # 3 dims 
            self.smooth_action   # 4 dims
        ]).astype(np.float32)
        
        obs = {"state": state_vec}
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        if self.use_camera or self.use_depth:
            cam_pos = pos + rot_mat @ [0.1, 0, 0]
            cam_target = pos + rot_mat @ [1.0, 0, 0]
            cam_up = rot_mat @ [0, 0, 1]
            
            view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), cam_up.tolist(), physicsClientId = self.env.CLIENT)
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0, physicsClientId = self.env.CLIENT)
            
            w, h, rgb, depth, seg = p.getCameraImage(64, 64, view, proj, renderer = p.ER_BULLET_HARDWARE_OPENGL, physicsClientId = self.env.CLIENT)
            
            img_stack = []
            if self.use_camera:
                rgb_norm = np.array(rgb, dtype = np.float32).reshape(64, 64, 4)[:, :, :3] / 255.0
                img_stack.append(np.transpose(rgb_norm, (2, 0, 1)))

            if self.use_depth:
                d_norm = np.array(depth, dtype = np.float32).reshape(1, 64, 64)
                img_stack.append(d_norm)

            obs["image"] = np.concatenate(img_stack, axis = 0)

        if self.use_lidar:
            ray_from, ray_to = [], []
            length = 5.0
            start_offset = 0.15 

            for i in range(self.lidar_rays):
                angle = 2 * np.pi * i / self.lidar_rays
                direction = rot_mat @ [np.cos(angle), np.sin(angle), 0]

                start = pos + direction * start_offset
                end = pos + direction * length
                
                ray_from.append(start.tolist())
                ray_to.append(end.tolist())
                
            results = p.rayTestBatch(ray_from, ray_to, physicsClientId = self.env.CLIENT)
            actual_ray_len = length - start_offset
            lidar_data = [start_offset + res[2] * actual_ray_len for res in results]
            obs["lidar"] = np.array(lidar_data, dtype = np.float32)
            
        return obs

    def close(self):
        self.env.close()