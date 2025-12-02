import numpy as np

class WaypointManager:
    def __init__(self):
        self.waypoints = []
        self.visual_ids = []

    def add_waypoint(self, x, y, z):
        waypoint = np.array([x, y, z])
        self.waypoints.append(waypoint)
        return waypoint
    
    def generate_obstacles(self, num_obstacles=6):
        if len(self.waypoints) < 2:
            return []

        obstacle_positions = []
        valid_segments = len(self.waypoints) - 1
        
        # Skip the first segment (0 -> 1) for safety
        start_segment = 1 if valid_segments > 1 else 0
        
        # Create a list of all possible segments indices
        available_indices = np.arange(start_segment, valid_segments)
        
        # Strategy: Sample indices to ensure spread
        if num_obstacles <= len(available_indices):
            # If we have enough segments, pick unique ones (No Overlap)
            chosen_indices = np.random.choice(available_indices, num_obstacles, replace=False)
        
        else:
            # If we have more obstacles than segments, we must reuse some
            # This ensures every segment gets at least one before we double up
            base_indices = np.tile(available_indices, int(np.ceil(num_obstacles / len(available_indices))))
            chosen_indices = base_indices[ : num_obstacles]

        for segment_idx in chosen_indices:
            p1 = self.waypoints[segment_idx]
            p2 = self.waypoints[segment_idx + 1]
            
            # Interpolate (30% to 70%)
            t = np.random.uniform(0.3, 0.7)
            pos = p1 + (p2 - p1) * t
            
            # Add Noise
            offset_x = np.random.uniform(-0.1, 0.1) 
            offset_y = np.random.uniform(-0.1, 0.1)
            offset_z = np.random.uniform(-0.3, 0.3)
            
            obstacle_positions.append(pos + np.array([offset_x, offset_y, offset_z]))
            
        return obstacle_positions

    def get_waypoints(self):
        return np.array(self.waypoints)

    def clear_waypoints(self):
        self.waypoints = []
        self.visual_ids = []

    def generate_hover_target(self, altitude):
        self.clear_waypoints()

        # We add the same point multiple times so the episode doesn't end 
        # instantly when the drone reaches it. It has to STAY there.
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        for _ in range(1000): 
            self.add_waypoint(x, y, altitude)
        
        print("Waypoint:", x, y, altitude)

        return self.get_waypoints()

    def generate_square_path(self, side_length, altitude): 
        self.clear_waypoints()
        half = side_length / 2
        coords = [
            (0, 0, altitude), (half, 0, altitude), (half, half, altitude),
            (0, half, altitude), (-half, half, altitude), (-half, 0, altitude),
            (-half, -half, altitude), (0, -half, altitude), (half, -half, altitude),
            (0, 0, altitude)
        ]

        for x, y, z in coords:
            self.add_waypoint(x, y, z)

        return self.get_waypoints()

    def generate_random_walk_path(self, num_waypoints = 10, min_dist = 1.5, max_dist = 4.0):
        self.clear_waypoints()

        current_pos = np.array([0.0, 0.0, 1.5])
        self.add_waypoint(*current_pos)

        current_heading = np.random.uniform(0, 2 * np.pi)

        MAX_TURN_ANGLE = np.pi / 3  # +/- 60 degrees (prevents 180 flips)
        Z_MIN, Z_MAX = 1.0, 3.0     # Safe flight corridor
        BOX_LIMIT = 10.0            # Keep drone within 10m of origin

        for _ in range(num_waypoints):
            angle_change = np.random.uniform(-MAX_TURN_ANGLE, MAX_TURN_ANGLE)
            current_heading += angle_change

            dist = np.random.uniform(min_dist, max_dist)

            dx = dist * np.cos(current_heading)
            dy = dist * np.sin(current_heading)
            dz = np.random.uniform(-0.5, 0.5) 

            new_pos = current_pos + np.array([dx, dy, dz])

            # If the new point is too far from origin, force the drone to turn back center
            if np.abs(new_pos[0]) > BOX_LIMIT or np.abs(new_pos[1]) > BOX_LIMIT:
                # Point towards origin + random noise
                direction_to_center = np.arctan2(-current_pos[1], -current_pos[0])
                current_heading = direction_to_center + np.random.uniform(-0.5, 0.5)

                # Recalculate with new heading to stay in bounds
                new_pos = current_pos + np.array([
                    dist * np.cos(current_heading),
                    dist * np.sin(current_heading),
                    dz
                ])

            new_pos[2] = np.clip(new_pos[2], Z_MIN, Z_MAX)

            self.add_waypoint(*new_pos)
            current_pos = new_pos

        return self.get_waypoints()