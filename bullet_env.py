import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BulletNavigationEnv(gym.Env):
    def __init__(self, waypoints, obstacles = [], use_camera = False, use_depth = False, 
                 use_lidar = False, use_obstacles = False, waypoint_threshold = 1.0, 
                 waypoint_bonus = 100.0, crash_penalty = -100.0, timeout_penalty = -10.0, 
                 step_reward = -0.1, episode_completion_reward = 100.0, max_dist_from_target = 10.0, 
                 action_smoothing = 0.75, action_limits = None,  gui = False,  show_waypoints = False):
        
        # Store a copy of the initial waypoints to detect the task type
        self.waypoints = waypoints
        self.obstacles = obstacles
        
        self.use_depth = use_depth
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_obstacles = use_obstacles

        self.show_waypoints = show_waypoints
        self.max_dist_from_target = max_dist_from_target
        
        # Reward Config
        self.step_reward = step_reward
        self.crash_penalty = crash_penalty
        self.waypoint_bonus = waypoint_bonus
        self.timeout_penalty = timeout_penalty
        self.episode_completion_reward = episode_completion_reward
        
        self.waypoint_threshold = waypoint_threshold

        self.lidar_rays = 360
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
        self.action_smoothing = action_smoothing  # Smoothing factor (0.1 = very smooth/sluggish, 0.9 = responsive/jerky)
        self.smooth_action = np.zeros(4)
        self.action_limits = np.array(action_limits) if action_limits is not None else np.array([1.0, 1.0, 1.0, 1.0])

        self.action_space = gym.spaces.Box(low = -1, high = 1, shape = (4,), dtype = np.float32)
        
        # State Dimension Calculation
        ## Target Direction (Unit Vector) -> 3
        ## Distance (Scalar)              -> 1
        ## Quaternion (Orientation)       -> 4
        ## Linear Velocity                -> 3
        ## Angular Velocity               -> 3
        ## Previous Action (Motor State)  -> 4 (self.action_space.shape[0])
        self.state_dim = 3 + 1 + 4 + 3 + 3 + self.action_space.shape[0]


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
        if self.use_obstacles:
            self._draw_obstacles()
        
        self.current_wp_idx = 0
        self.prev_action = np.zeros(4) 
        self.smooth_action = np.zeros(4)
        self.target_pos = self.waypoints[self.current_wp_idx]
        self.prev_dist = np.linalg.norm(self._get_drone_pos() - self.target_pos)

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

    def _draw_obstacles(self):
        if not self.show_waypoints:
            return
        
        if p.getConnectionInfo(physicsClientId = self.env.CLIENT)["connectionMethod"] == p.GUI:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            for pos in self.obstacles:
                p.loadURDF("cube.urdf", pos, globalScaling = 0.6, useFixedBase = True, physicsClientId = self.env.CLIENT)

    def step(self, action):  
        self.smooth_action = self.action_smoothing * action + (1 - self.action_smoothing) * self.smooth_action    
        scaled_action = self.smooth_action * self.action_limits

        pos, quat = p.getBasePositionAndOrientation(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        lin_vel, ang_vel = p.getBaseVelocity(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)

        current_pos = np.array(pos)
        lin_vel_np = np.array(lin_vel)

        r = R.from_quat(quat)

        action_body = np.array(scaled_action[:3])
        target_v_world = r.apply(action_body)

        # Hard-Coded Yaw
        diff_vec = self.target_pos - current_pos
        desired_yaw = np.arctan2(diff_vec[1], diff_vec[0])

        # RL Yaw Rate
        # target_yaw_rate = scaled_action[3] 
        
        # Compute RPMs using DSLPIDControl
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep = 1.0 / self.env.CTRL_FREQ,

            cur_pos = current_pos,
            cur_quat = np.array(quat),
            cur_vel = lin_vel_np,
            cur_ang_vel = np.array(ang_vel),

            target_pos = current_pos, # Keep current pos target to force velocity tracking
            target_vel = target_v_world,

            # Hard-Coded Yaw
            target_rpy = np.array([0, 0, desired_yaw]),
            target_rpy_rates = np.array([0, 0, 0])
            
            # RL Yaw Rate
            # target_rpy = np.array([0, 0, 0]),
            # target_rpy_rates = np.array([0, 0, target_yaw_rate])
        )
        
        # Step the environment with calculated RPMs
        self.env.step(rpm.reshape(1, 4))

        # We must re-read position to calculate reward for the NEW state
        pos, quat = p.getBasePositionAndOrientation(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        lin_vel, ang_vel = p.getBaseVelocity(self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
        
        current_pos = np.array(pos)
        lin_vel_np = np.array(lin_vel)

        obs = self._get_observation()

        reward = 0
        terminated = False
        truncated = False
        
        reward += (self.step_reward)
        
        dist = np.linalg.norm(current_pos - self.target_pos)

        progress = self.prev_dist - dist
        reward += 30.0 * progress

        # Acceleration/Jerk Penalty: Penalize changing motor commands too quickly
        diff_action = action - self.prev_action
        reward -= 0.01 * np.linalg.norm(diff_action) ** 2

        # High Velocity Penalty
        reward -= 0.05 * np.linalg.norm(lin_vel) ** 2
        reward -= 0.005 * np.linalg.norm(ang_vel) ** 2

        # Alignment Penalty
        ## Get the direction vector to the target (Normalized)
        target_vec = self.target_pos - current_pos
        if dist > 1e-6:
            target_dir = target_vec / dist
        else:
            target_dir = np.zeros(3)

        # Get the drone's actual forward direction in World coordinates
        forward_world = r.apply([1, 0, 0]) # The body X-axis rotated to World frame

        # Calculate alignment (Dot Product)
        # We only care about 2D alignment (Yaw) for navigation, so ignore Z
        alignment = np.dot(forward_world[:2], target_dir[:2])

        # alignment ranges from -1.0 to 1.0.
        # If you want it to be a pure penalty for misalignment:
        # Penalty is 0 when aligned, and high when facing away.
        heading_penalty = (1.0 - alignment) 
        reward -= 0.1 * heading_penalty 

        # OR, if you prefer positive reinforcement (Reward for facing correctly):
        # reward += 0.1 * alignment

        # print(
        #     self.step_reward,
        #     30.0 * progress,
        #     -0.01 * np.linalg.norm(diff_action) ** 2,
        #     -0.05 * np.linalg.norm(lin_vel) ** 2,
        #     0.1 * heading_penalty 
        # )

        # print("Alignment:", alignment)

        if dist < self.waypoint_threshold:
            reward += self.waypoint_bonus
            self.current_wp_idx += 1

            if self.current_wp_idx >= len(self.waypoints):
                print("--- Course Complete! ---")
                reward += self.episode_completion_reward
                terminated = True

            else:
                self.target_pos = self.waypoints[self.current_wp_idx]
                print(f"--- Reached Waypoint {self.current_wp_idx} ---")
                self.prev_dist = np.linalg.norm(current_pos - self.target_pos)

        contact_points = p.getContactPoints(bodyA = self.env.DRONE_IDS[0], physicsClientId = self.env.CLIENT)
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

        lin_vel_scaled = np.clip(lin_vel_body, -2.0, 2.0) / 2.0
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
            near_plane = 0.1
            far_plane = 20.0 # Reduced from 100.0 to give better resolution for indoor flight

            cam_pos = pos + rot_mat @ [0.1, 0, 0]
            cam_target = pos + rot_mat @ [1.0, 0, 0]

            p.addUserDebugLine(cam_pos, cam_target, [0, 0, 1], lineWidth = 2, lifeTime = 0.1, physicsClientId = self.env.CLIENT)

            cam_up = rot_mat @ [0, 0, 1]
            
            view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), cam_up.tolist(), physicsClientId = self.env.CLIENT)
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0, physicsClientId = self.env.CLIENT)
            
            w, h, rgb, depth, seg = p.getCameraImage(64, 64, view, proj, renderer = p.ER_BULLET_HARDWARE_OPENGL, physicsClientId = self.env.CLIENT)
            
            img_stack = []
            if self.use_camera:
                rgb_norm = np.array(rgb, dtype = np.float32).reshape(64, 64, 4)[:, :, :3] / 255.0
                img_stack.append(np.transpose(rgb_norm, (2, 0, 1)))

            if self.use_depth:
                # Reshape raw OpenGL buffer
                depth = np.array(depth, dtype = np.float32).reshape(64, 64)

                # Convert to Linear Depth (Meters)
                # Formula: 2 * far * near / (far + near - (2 * depth - 1) * (far - near))
                # Note: PyBullet depth buffer is range [0, 1]
                depth_linear = (2.0 * near_plane * far_plane) / (far_plane + near_plane - (2.0 * depth - 1.0) * (far_plane - near_plane))

                depth_norm = depth_linear / far_plane
                depth_norm = np.clip(depth_norm, 0.0, 1.0)

                d_final = depth_norm.reshape(1, 64, 64)
                img_stack.append(d_final)

            obs["image"] = np.concatenate(img_stack, axis = 0)

            # print(f"Shape: {obs.get('image').shape}")

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