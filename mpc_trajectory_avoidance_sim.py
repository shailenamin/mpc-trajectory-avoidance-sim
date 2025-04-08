import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from matplotlib.animation import FuncAnimation

class MPCTrajectoryTracker:
    def __init__(self):
        # Simulation parameters
        self.dt = 0.1  # Time step
        self.T = 20    # Total simulation time
        self.N = int(self.T / self.dt)  # Number of time steps

        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.control_horizon = 5  # Shorter control horizon for efficiency
        self.max_iter = 100  # Max optimization iterations

        # Robot constraints
        self.max_vel = 1.5  # Maximum velocity
        self.max_acc = 1.0   # Maximum acceleration

        # State: [x, y, vx, vy]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        # Trajectory storage
        self.goal_traj = []
        self.obs_traj = []
        self.actual_traj = [self.state[:2].copy()]

        # Obstacle parameters
        self.obstacle_radius = 0.5
        self.safety_margin = 0.3  # Additional safety margin

        # Cost function weights
        self.w_goal = 1.0      # Goal tracking weight
        self.w_control = 0.1   # Control effort weight
        self.w_obs = 1000.0    # Obstacle avoidance weight
        self.w_vel = 0.05      # Velocity tracking weight

        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.animation = None

    def desired_path(self, t):
        """Moving figure-8 trajectory"""
        x = 4 * np.sin(0.2 * t)
        y = 4 * np.sin(0.2 * t) * np.cos(0.2 * t)
        return np.array([x, y])

    def obstacle_motion(self, t):
        """Obstacle moves in a circle"""
        x = 2.5 + 1.5 * np.cos(0.3 * t)
        y = 2.5 + 1.5 * np.sin(0.3 * t)
        return np.array([x, y])

    def dynamics(self, state, u):
        """Robot dynamics with velocity clipping"""
        x, y, vx, vy = state
        ax, ay = u

        # Clip acceleration first
        ax = np.clip(ax, -self.max_acc, self.max_acc)
        ay = np.clip(ay, -self.max_acc, self.max_acc)

        # Update velocity with clipping
        vx_new = np.clip(vx + ax * self.dt, -self.max_vel, self.max_vel)
        vy_new = np.clip(vy + ay * self.dt, -self.max_vel, self.max_vel)

        # Update position
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt

        return np.array([x_new, y_new, vx_new, vy_new])

    def predict_trajectory(self, state, U):
        """Predict future trajectory given control sequence"""
        traj = []
        x = state.copy()
        for u in U:
            x = self.dynamics(x, u)
            traj.append(x[:2])
        return np.array(traj)

    def cost_function(self, U_flat, state, goal_seq, obs_seq):
        """Cost function for MPC optimization"""
        U = U_flat.reshape(-1, 2)
        x = state.copy()
        total_cost = 0.0

        # Use shorter control horizon but full prediction horizon
        for i in range(self.horizon):
            if i < self.control_horizon:
                u = U[i]
            else:
                u = np.zeros(2)  # Zero control beyond control horizon

            x = self.dynamics(x, u)
            goal = goal_seq[i]
            obs = obs_seq[i]

            # Goal tracking cost
            dist_to_goal = np.linalg.norm(x[:2] - goal)
            total_cost += self.w_goal * dist_to_goal**2

            # Velocity tracking cost (try to match desired velocity)
            if i < len(goal_seq) - 1:
                desired_vel = (goal_seq[i+1] - goal_seq[i]) / self.dt
                vel_diff = np.linalg.norm(x[2:] - desired_vel)
                total_cost += self.w_vel * vel_diff**2

            # Control effort cost
            if i < self.control_horizon:
                total_cost += self.w_control * np.linalg.norm(u)**2

            # Obstacle avoidance cost (smooth penalty function)
            dist_to_obs = np.linalg.norm(x[:2] - obs)
            min_dist = self.obstacle_radius + self.safety_margin
            if dist_to_obs < min_dist:
                total_cost += self.w_obs * (min_dist - dist_to_obs)**2

        return total_cost

    def run_mpc(self):
        """Run MPC simulation"""
        start_time = time.time()

        for t_step in range(self.N):
            t = t_step * self.dt

            # Generate reference trajectories
            goal_seq = np.array([self.desired_path(t + i*self.dt) for i in range(self.horizon+1)])
            obs_seq = np.array([self.obstacle_motion(t + i*self.dt) for i in range(self.horizon)])

            # Store for visualization
            self.goal_traj.append(goal_seq[0])
            self.obs_traj.append(obs_seq[0])

            # Initial guess (warm start) - use previous solution shifted
            if t_step == 0:
                U0 = np.zeros((self.control_horizon, 2)).flatten()
            else:
                U0 = np.roll(res.x, -2)
                U0[-2:] = 0  # Fill the end with zeros

            # Optimize control sequence
            bounds = [(-self.max_acc, self.max_acc)] * self.control_horizon * 2
            res = minimize(
                self.cost_function,
                U0,
                args=(self.state, goal_seq, obs_seq),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': self.max_iter, 'disp': False}
            )

            # Apply first control input
            u_opt = res.x[:2] if res.success else np.zeros(2)
            self.state = self.dynamics(self.state, u_opt)
            self.actual_traj.append(self.state[:2].copy())

            # Early termination if close to final goal
            if np.linalg.norm(self.state[:2] - goal_seq[-1]) < 0.2:
                print(f"Reached goal at t = {t:.1f} seconds")
                break

        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

        # Convert to numpy arrays
        self.actual_traj = np.array(self.actual_traj)
        self.goal_traj = np.array(self.goal_traj)
        self.obs_traj = np.array(self.obs_traj)

    def plot_results(self):
        """Plot the results"""
        self.ax.plot(self.goal_traj[:, 0], self.goal_traj[:, 1], 'r--', label='Desired Path')
        self.ax.plot(self.actual_traj[:, 0], self.actual_traj[:, 1], 'b-', linewidth=2, label='Robot Path')
        self.ax.plot(self.obs_traj[:, 0], self.obs_traj[:, 1], 'k:', label='Obstacle Path')

        # Add start and end markers
        self.ax.scatter(self.actual_traj[0, 0], self.actual_traj[0, 1],
                       color='green', label='Start', s=100, zorder=5)
        self.ax.scatter(self.goal_traj[-1, 0], self.goal_traj[-1, 1],
                       color='purple', label='Final Goal', s=100, zorder=5)

        # Add obstacle visualization
        for obs_pos in self.obs_traj[::10]:  # Plot every 10th obstacle position
            circle = plt.Circle(obs_pos, self.obstacle_radius,
                              color='gray', alpha=0.3)
            self.ax.add_patch(circle)

        self.ax.set_title("Improved MPC Trajectory Tracking with Obstacle Avoidance")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.axis("equal")
        self.ax.grid(True)
        self.ax.legend()
        plt.tight_layout()
        plt.savefig("improved_mpc_trajectory.png", dpi=300)
        plt.show()

    def animate_results(self):
        """Create an animation of the results"""
        self.fig, self.ax = plt.subplots(figsize=(10, 7))

        def init():
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.grid(True)
            return []

        def update(frame):
            self.ax.clear()
            self.ax.set_title(f"MPC Trajectory Tracking - Time: {frame*self.dt:.1f}s")

            # Plot full trajectories up to current frame
            self.ax.plot(self.goal_traj[:frame, 0], self.goal_traj[:frame, 1], 'r--', label='Desired Path')
            self.ax.plot(self.actual_traj[:frame+1, 0], self.actual_traj[:frame+1, 1], 'b-', label='Robot Path')
            self.ax.plot(self.obs_traj[:frame, 0], self.obs_traj[:frame, 1], 'k:', label='Obstacle Path')

            # Current positions
            self.ax.scatter(self.goal_traj[frame, 0], self.goal_traj[frame, 1],
                           color='red', s=50, label='Current Goal')
            self.ax.scatter(self.actual_traj[frame+1, 0], self.actual_traj[frame+1, 1],
                           color='blue', s=100, label='Robot')

            # Obstacle with safety margin
            obs_circle = plt.Circle(self.obs_traj[frame], self.obstacle_radius,
                                  color='gray', alpha=0.5)
            self.ax.add_patch(obs_circle)

            # Prediction horizon visualization
            if frame % 5 == 0:  # Only show predictions every 5 frames
                goal_seq = np.array([self.desired_path(frame*self.dt + i*self.dt)
                                   for i in range(self.horizon)])
                obs_seq = np.array([self.obstacle_motion(frame*self.dt + i*self.dt)
                                  for i in range(self.horizon)])
                self.ax.plot(goal_seq[:, 0], goal_seq[:, 1], 'r.', markersize=5, alpha=0.5)
                self.ax.plot(obs_seq[:, 0], obs_seq[:, 1], 'k.', markersize=5, alpha=0.5)

            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.legend(loc='upper right')
            self.ax.grid(True)

            return []

        frames = min(len(self.actual_traj)-1, len(self.goal_traj))
        self.animation = FuncAnimation(self.fig, update, frames=frames,
                                      init_func=init, blit=True, interval=50)
        plt.show()
        return self.animation

# Run the simulation
if __name__ == "__main__":
    tracker = MPCTrajectoryTracker()
    tracker.run_mpc()
    tracker.plot_results()
    # Uncomment to create animation (requires ffmpeg)
    # tracker.animate_results()