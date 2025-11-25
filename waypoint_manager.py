import numpy as np

class WaypointManager:
    def __init__(self):
        self.waypoints = []
        self.visual_ids = []

    def add_waypoint(self, x, y, z):
        waypoint = np.array([x, y, z])
        self.waypoints.append(waypoint)
        return waypoint

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

    def generate_random_walk_path(self, num_waypoints = 10, max_step_dist = 5.0):
        self.clear_waypoints()

        last_wp = np.array([0.0, 0.0, 1.5]) # Start lower
        self.add_waypoint(*last_wp)
        
        for _ in range(num_waypoints - 1):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(2.0, max_step_dist) 

            new_x = last_wp[0] + dist * np.cos(angle)
            new_y = last_wp[1] + dist * np.sin(angle)
            new_z = np.clip(last_wp[2] + np.random.uniform(-1, 1), 1.0, 3.0) # Keep altitude manageable
            
            self.add_waypoint(new_x, new_y, new_z)
            last_wp = np.array([new_x, new_y, new_z])
            
        return self.get_waypoints()