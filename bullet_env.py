import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class BulletNavigationEnv(gym.Env):
    def __init__(self, waypoints, use_camera = False, use_depth = False, use_lidar = False, 
                 use_obstacles = False, waypoint_threshold = 1.0, waypoint_bonus = 100.0,
                 crash_penalty = 100.0, timeout_penalty = 10.0, per_step_penalty = -0.1,
                 max_dist_from_target = 10.0, gui = False, show_waypoints = False):
        
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
        
        # Visualization IDs
        self.marker_ids = [] 
        self.debug_line_ids = []

        # PyBullet Drone Init (Using CtrlAviary for correct physics stepping)
        self.env = CtrlAviary(
            drone_model = DroneModel.CF2X,
            num_drones = 1,
            neighbourhood_radius = 10,
            physics = Physics.PYB,
            gui = gui,
            pyb_freq = 240,
            ctrl_freq = 240 #48
        )
        
        # Action & Observation Spaces
        self.action_space = gym.spaces.Box(low = -1, high = 1, shape = (4,), dtype = np.float32)
        self.state_dim = 16
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

        if self.use_obstacles:
            self._spawn_obstacles()

        return self._get_observation(), {}

    def _draw_waypoints(self):
        if not self.show_waypoints:
            return
        
        if p.getConnectionInfo(physicsClientId = self.env.CLIENT)["connectionMethod"] == p.GUI:
            self.marker_ids = []
            self.debug_line_ids = []

            for i, wp in enumerate(self.waypoints):
                print("Waypoint", i, wp)

                visual_shape_id = p.createVisualShape(
                    shapeType = p.GEOM_SPHERE,
                    radius = 0.1,             # Slightly larger to be visible
                    rgbaColor = [0, 1, 0, 1], # Green
                    physicsClientId = self.env.CLIENT
                )
                
                body_id = p.createMultiBody(
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
                        lineColorRGB = [1, 0, 0], # Red
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
        hover_rpm = 31700
        thrust = action[3]
        base_rpm = hover_rpm * (thrust * 4000)
        
        mix_matrix = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]])
        mix = mix_matrix @ action[:3]
        pwms = np.clip(base_rpm + mix * 4000, 0, 60000)
        
        self.env.step(pwms.reshape(1, 4))
        
        # 2. Get Obs & Reward
        obs = self._get_observation()
        reward = self.per_step_penalty
        
        current_pos = obs["state"][:3]
        dist = np.linalg.norm(current_pos - self.target_pos)
        
        reward += (self.prev_dist - dist) * 10.0
        self.prev_dist = dist
        
        terminated = False
        truncated = False

        # Terminate if too far from the CURRENT target waypoint
        if dist > self.max_dist_from_target:
            reward -=  self.timeout_penalty
            terminated = True

        if dist < self.waypoint_threshold:
            reward +=  self.waypoint_bonus
            self.current_wp_idx +=  1
            if self.current_wp_idx >=  len(self.waypoints):
                print("--- Course Complete! ---")
                terminated = True

            else:
                self.target_pos = self.waypoints[self.current_wp_idx]
                print(f"--- Reached Waypoint {self.current_wp_idx} ---")
                current_pos = obs["state"][:3]
                self.prev_dist = np.linalg.norm(current_pos - self.target_pos)

        if current_pos[2] < 0.1: 
            reward -= self.crash_penalty
            terminated = True
            
        if abs(current_pos[0]) > 20 or abs(current_pos[1]) > 20 or current_pos[2] > 20:
            reward -= self.crash_penalty
            terminated = True

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
        
        target_vec = self.target_pos - pos
        state_vec = np.concatenate([pos, quat, lin_vel, ang_vel, target_vec]).astype(np.float32)
        
        obs = {"state": state_vec, "image": None, "lidar": None}
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