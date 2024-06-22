import numpy as np
import matplotlib.pyplot as plt

U = 200  # meters per second
g = 9.8  # meters per second squared
S0 = 0   # meters

class TrajectoryTracker():
    def __init__(self,n_samples):
        self.n_samples= n_samples
        np.random.seed(2)
        
        # test data for the variance-covariance matrix
        self.sample_V = np.round([U - g * t * 0.2 for t in range(self.n_samples)], 2)
        self.sample_S = np.round([S0 + t * self.sample_V[t] * 0.2 - 0.5 * (t * 0.2)**2 * g for t in range(self.n_samples)])
        self.test_data = np.vstack((self.sample_S, self.sample_V))
        self.test_data = np.transpose(self.test_data)
        
        self.dt = 1
        self.initial_s = S0
        self.initial_v = U
        self.initial_state = np.array([[self.initial_s], [self.initial_v]])
        self.g = 9.8
        self.A = np.eye(2, dtype=np.float32)
        self.A[0][1] = self.dt
        self.B = np.array([[0.5 * self.dt**2], [self.dt]], dtype=np.float32)
        self.U = np.array([-self.g], dtype=np.float32)
        self.W = np.zeros((2, 1), dtype=np.float32)
        self.Q = np.eye(2, dtype=np.float32) * 0.1
        self.H = np.eye(2, dtype=np.float32)
        self.R = np.array([[100, 0], [0, 3.2]], dtype=np.float32)
        self.C = np.eye(2, dtype=np.float32)
        self.Z = np.zeros((2, 1), dtype=np.float32)
        self.prev_cov = self.compute_var_cov_matrix(self.test_data)

    def create_measurement_data(self):
        n = np.random.randint(-20, 20, size=self.sample_S.shape)
        n = np.sin(n)
        self.mea_s = self.sample_S + n*0.1
        self.mea_v = self.sample_V + np.random.randint(-10, 10, size=self.sample_V.shape)
        self.measurement = np.vstack((self.mea_s, self.mea_v))
        self.measurement = np.transpose(self.measurement)
        return self.measurement
    
    def compute_var_cov_matrix(self, data):
        cov_matrix = np.cov(data.T)
        return cov_matrix
    
    def state_estimation(self, initial_state):
        state_estimate = np.dot(self.A, initial_state) + np.dot(self.B, self.U) + self.W
        return state_estimate
    
    def state_covariance_estimation(self, prev_cov):
        pred_cov = np.dot(np.dot(self.A, prev_cov), np.transpose(self.A)) + self.Q
        return pred_cov

    def compute_kalman_gain(self, cov_matrix):
        S = np.dot(np.dot(self.H, cov_matrix), self.H.T) + self.R
        Kg = np.dot(np.dot(cov_matrix, self.H.T), np.linalg.inv(S))
        return Kg
    
    def make_observations(self, prev_obs):
        new_obs = np.dot(self.C, prev_obs) + self.Z
        return new_obs
    
    def current_state(self, prev_state, obs, k):
        curr_state = prev_state + np.dot(k, obs - np.dot(self.H, prev_state))
        return curr_state
    
    def update_process_cov(self, prev_cov, k):
        new_cov = np.dot((np.eye(2, dtype=np.float32) - np.dot(k, self.H)), prev_cov)
        return new_cov
    
    def plot_states(self, pred_states, meas_states, curr_states):
        pred_positions = [state[0][0] for state in pred_states]
        pred_velocities = [state[1][0] for state in pred_states]
        meas_positions = [state[0] for state in meas_states]
        meas_velocities = [state[1] for state in meas_states]
        curr_positions = [state[0][0] for state in curr_states]
        curr_velocities = [state[1][0] for state in curr_states]

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(pred_positions, label='Predicted Position', marker='o', color="blue")
        plt.plot(meas_positions, label='Measured Position', marker='*', color="red")
        plt.plot(curr_positions, label='Filtered Position', marker='x', color="green")
        plt.title(f'Position Tracking for {self.n_samples} samples')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(pred_velocities, label='Predicted Velocity', marker='o', color="blue")
        plt.plot(meas_velocities, label='Measured Velocity', marker='*', color="red")
        plt.plot(curr_velocities, label='Filtered Velocity', marker='x', color="green")
        plt.title(f'Velocity Tracking for {self.n_samples} samples')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def start_tracking(self):
        pred_val = []
        curr_val = []
        meas_val = []
        mea = self.create_measurement_data()
        for i in range(self.n_samples):
            state_pred = self.state_estimation(self.initial_state)
            state_cov = self.state_covariance_estimation(self.prev_cov)
            K = self.compute_kalman_gain(state_cov)
            old_obs = np.array([[mea[i][0]], [mea[i][1]]])
            new_obs = self.make_observations(old_obs)
            curr_state = self.current_state(state_pred, new_obs, K)
            new_cov = self.update_process_cov(state_cov, K)
            curr_val.append(curr_state)
            pred_val.append(state_pred)
            meas_val.append([mea[i][0], mea[i][1]])
            self.initial_state = curr_state
            self.prev_cov = new_cov
        self.plot_states(pred_val, meas_val, curr_val)
            
def main(n_samples):
    tracker = TrajectoryTracker(n_samples)
    tracker.start_tracking()
    
if __name__ == "__main__":
    n = int(input("Enter number of samples: "))
    main(n)
