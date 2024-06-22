import numpy as np
import matplotlib.pyplot as plt
U = 200 #meters per seconds
g = 9.8 #meters per seconds squared
S0 = 0 #meters

class TrajectoryTracker():
    def __init__(self):
        np.random.seed(42)
        
        #test data for the variance-covarianxce matrix
        self.sample_V = np.round([U-g*t for t in range(40)],2)
        self.sample_S = np.round([(S0 + t*self.sample_V[t-1] - 0.5*t*g) for t in range(40)])
        self.test_data = np.vstack((self.sample_S,self.sample_V))
        self.test_data = np.transpose(self.test_data)
        
        self.dt = 0.1
        self.initial_s = 0
        self.initial_v = 0
        self.initial_state = np.array([[self.initial_s],[self.initial_v]])
        self.g = 9.8
        self.A = np.eye(2,dtype=np.float32)
        self.A[0][1] = self.dt
        self.B = np.array([[0.5*self.dt**2],[self.dt]],dtype=np.float32)
        self.U = np.array(-self.g,dtype=np.float32)
        self.W = np.zeros((2,1),dtype=np.float32)
        self.Q = np.eye(2,dtype=np.float32) * 0.1
        self.H = np.eye(2,dtype=np.float32)
        self.R = np.array([[100,0],[0,3.2]],dtype=np.float32)
        # self.R = 0
        self.C = np.eye(2,dtype=np.float32)
        self.Z = 0
        self.prev_cov = self.compute_var_cov_matrix(self.test_data)
        #create noise
    def create_measurement_data(self):
        self.mea_s = self.sample_S + np.random.randint(5,15)
        self.mea_v = self.sample_V + np.random.randint(0,6)
        self.measurement = np.vstack((self.mea_s,self.mea_v))
        self.measurement = np.transpose(self.measurement)
        return self.measurement
    
    def compute_var_cov_matrix(self,data):
        ave_data = np.mean(data)
        deviation =  data - ave_data
        cov_matrix = np.dot(deviation.T,deviation)/(data.shape[0] -1)
        return cov_matrix
    
    def state_estimation(self,initial_state):
        state_estimate = np.dot(self.A,initial_state) + np.dot(self.B,self.U) + self.W
        return state_estimate
    
    def state_covariance_estimation(self,prev_cov):
        pred_cov = np.dot(np.dot(self.A,prev_cov),np.transpose(self.A)) + self.Q
        return pred_cov

    def compute_kalman_gain(self, cov_matrix):
        S = np.dot(np.dot(self.H, cov_matrix), self.H.T) + self.R
        Kg = np.dot(np.dot(cov_matrix, self.H.T), np.linalg.inv(S))
        return Kg
    
    def make_observations(self,prev_obs):
        new_obs = np.dot(self.C,prev_obs) + self.Z
        return new_obs
    
    def current_state(self,prev_state,obs,k):
        curr_state = prev_state + np.dot(k,obs - np.dot(self.H,prev_state))
        return curr_state
    
    def update_process_cov(self,prev_cov,k):
        new_cov = np.dot((np.eye(2,dtype=np.float32) - np.dot(k,self.H)),prev_cov)
        return new_cov
    
    def plot_states(self, pred_states, curr_states):
        pred_positions = [state[0][0] for state in pred_states]
        pred_velocities = [state[1][0] for state in pred_states]
        curr_positions = [state[0][0] for state in curr_states]
        curr_velocities = [state[1][0] for state in curr_states]

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(pred_positions, label='Predicted Position', marker='o',color="green")
        plt.plot(curr_positions, label='Current Position', marker='*',color="red")
        plt.title('Position Tracking')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)

        # Ensure auto-scaling
        plt.autoscale(True)

        plt.subplot(1, 2, 2)
        plt.plot(pred_velocities, label='Predicted Velocity', marker='o',color="green")
        plt.plot(curr_velocities, label='Current Velocity', marker='*', color="red")
        plt.title('Velocity Tracking')
        plt.xlabel('Time Step')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)

        # Ensure auto-scaling
        plt.autoscale(True)

        plt.tight_layout()
        plt.show()

    
    def start_tracking(self):
        pred_val = []
        curr_val = []
        for i in range(len(self.sample_S)):
            state_pred = self.state_estimation(self.initial_state)
            state_cov = self.state_covariance_estimation(self.prev_cov)
            K = self.compute_kalman_gain(state_cov)
            old_obs = [[self.sample_S[i]],[self.sample_V[i]]]
            new_obs = self.make_observations(old_obs)
            curr_state = self.current_state(state_pred,new_obs,K)
            new_cov = self.update_process_cov(state_cov,K)
            # print(curr_state)
            # print()
            curr_val.append(curr_state)
            pred_val.append(state_pred)
            self.initial_state = curr_state
            self.prev_cov = new_cov
        self.plot_states(pred_val,curr_val)
            
def main():
    tracker = TrajectoryTracker()
    tracker.start_tracking()
    
if __name__ == "__main__":
    main()