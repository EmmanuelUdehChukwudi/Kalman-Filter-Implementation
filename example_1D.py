import numpy as np
from matplotlib import pyplot as plt
import time

"""
    
    This is a simple Implementation of a 1-D Kalman filter.
    This system is for Tracking the position(x) and velocity(v) of an object
"""

        # Initial states
X0 = 0
V0 = 1
dt = 1
initial_X = np.array([[X0],[V0]],dtype=np.float32)

        # Initial Uncertainty (Variance)

var_x = 1.0
var_v = 1.0
initial_P = np.array([[1,0],[0,1]],dtype=np.float32)
Q = np.array([[0.1,0],[0,0.1]],dtype=np.float32) # process noise cov matrix
U = 0 # control variable matrix

        # Measurements
        
X = np.array([1.1, 2.0, 2.9, 3.7, 5.0, 5.9, 7.2, 7.8, 9.0, 9.9,
              11.0, 12.0, 13.1, 14.0, 15.2, 16.0, 17.3, 18.1, 19.0, 20.2,
              21.1, 22.0, 23.2, 24.0, 25.3, 26.0, 27.1, 28.2, 29.1, 30.0,
              31.2, 32.0, 33.1, 34.0, 35.3, 36.0, 37.1, 38.0, 39.2, 40.0,
              41.1, 42.0, 43.3, 44.0, 45.2, 46.0, 47.1, 48.0, 49.2, 50.0], dtype=np.float32)  #  position measurements

V = np.array([0.9, 1.1, 1.0, 0.8, 1.2, 0.9, 1.1, 0.9, 1.2, 1.0,
              0.8, 1.1, 1.0, 1.2, 0.9, 1.1, 1.0, 0.9, 1.2, 1.0,
              0.9, 1.0, 1.1, 0.9, 1.2, 1.0, 0.9, 1.1, 1.0, 1.2,
              0.9, 1.1, 1.0, 1.2, 0.9, 1.0, 1.1, 0.9, 1.2, 1.0,
              0.8, 1.1, 1.0, 1.2, 0.9, 1.1, 1.0, 0.9, 1.2, 1.0], dtype=np.float32)  #  velocity measurements

R = np.array([[0.005,0.005],[0.005,0.005]],dtype=np.float32) #sensor noise cov
Z = np.array([[0.1],[0.05]],dtype=np.float32) # measurement noise (uncertainty)

        # state Transition matrices
A = np.array([[1,dt],[0,1]],dtype=np.float32) 
B = 0
H = np.eye(2,dtype=np.float32)
W = 0
C = np.eye(2,dtype=np.float32)

#Predict state
def predict_state(start_state):
        prev_state = np.dot(A,start_state) + np.dot(B,U) + W
        return prev_state

# process covariance matrix
def pred_process_cov(start_process_cov):
        prev_process_cov = np.dot(np.dot(A,start_process_cov),np.transpose(A)) + Q
        return prev_process_cov


# calculate kalman gain
def compute_kalman_gain(process_cov,sensor_noise_cov):
        k = np.dot(process_cov,np.transpose(H)) / (np.dot(np.dot(H,process_cov),np.transpose(H)) + sensor_noise_cov)
        return k

# New Observation(reading)
def new_observation(prev_readings):
        new_observation = np.dot(C,prev_readings) + Z
        return new_observation

# Calculate current state
def current_state(prev_state,new_observation,k):
        new_state = prev_state + np.dot(k,(new_observation-np.dot(H,prev_state)))
        return new_state

# update process cov matrix
def update_process_cov(prev_process_cov,k):
        r = np.eye(2,dtype=np.float32) - np.dot(k,H)
        new_process_cov = np.dot(r,prev_process_cov)
        return new_process_cov

# Function to plot the position and velocity
def plot_states(pred_states, curr_states, X0, V0):
    pred_positions = [state[0] for state in pred_states]
    pred_velocities = [state[1] for state in pred_states]
    curr_positions = [state[0] for state in curr_states]
    curr_velocities = [state[1] for state in curr_states]

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(pred_positions, label='Predicted Position', marker='o')
    plt.plot(curr_positions, label='Current Position', marker='x')
    plt.axhline(y=X0, color='r', linestyle='--', label='Initial Position')
    plt.title('Position Tracking')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(pred_velocities, label='Predicted Velocity', marker='o')
    plt.plot(curr_velocities, label='Current Velocity', marker='x')
    plt.axhline(y=V0, color='r', linestyle='--', label='Initial Velocity')
    plt.title('Velocity Tracking')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.legend()

    plt.tight_layout()
    plt.show()



def main():
        global initial_X,initial_P,Q,X0,V0
        pred_states = [initial_X]
        curr_states = [initial_X]
        for i in range(len(X)):
                prev_pred_state = predict_state(start_state=initial_X)
                prev_pro_cov = pred_process_cov(initial_P)
                kg = compute_kalman_gain(prev_pro_cov,sensor_noise_cov=R)
                new_obs = new_observation(prev_readings=np.array([[X[i]],[V[i]]]))
                curr_state = current_state(prev_state=prev_pred_state,new_observation=new_obs,k=kg)
                new_proceess_cov = update_process_cov(prev_pro_cov,k = kg)
                pred_states.append(prev_pred_state)
                curr_states.append(curr_state)
                print(np.matrix(np.round(curr_state,3)))
                # print(kg)
                print()
                initial_P = new_proceess_cov
                initial_X = curr_state
                # time.sleep(1.0)
        plot_states(pred_states, curr_states, X0, V0)
if __name__ == "__main__":
        main()

