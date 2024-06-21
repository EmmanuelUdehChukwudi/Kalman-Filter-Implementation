"""
    This Code implements tracking of the horizontal and vertical distance of a projectile in 2D (x,y) and the velocities in x and y.
    It assumes angle of projection to be 90 degrees.
"""
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(42)

class ObjectTracker():
    def __init__(self):
        self.start_state = np.array([[0],[0],[0],[0]]) 
        self.dt = 1
        self.g = -9.8
        self.start_cov = np.zeros(4,dtype=np.float32)
        self.X = np.arange(start=0,step=1.5,stop=75,dtype=np.float32)
        self.Y = (self.X -2)**2 + 1.226*self.X - 1.33
        self.Vx = [0 for x in range(50)]
        self.Vy = [self.g*(self.dt+u) for u in range(50)]
       
        self.R = np.array([[0.0001,0.0001,0.0001,0.0001],[0.0001,0.0001,0.0001,0.0001],
                           [0.0001,0.0001,0.0001,0.0001],[0.0001,0.0001,0.0001,0.0001]])
        self.B = np.array([[0.5*self.dt**2,0],[0,0.5*self.dt**2],[self.dt,0],[0,self.dt]],dtype=np.float32)
        self.U = np.array([[0],[self.g]],dtype=np.float32)
        self.A = np.eye(4,dtype=np.float32)
        self.A[0][2] = self.dt
        self.A[1][3] = self.dt
        self.C = np.eye(4,dtype=np.float32)
        self.Z = 0
        self.H = np.eye(4,dtype=np.float32)
        
        self.positions = []
        self.velocities = []
        
        
    def predict_state(self,start_state):
        self.state_pred = np.dot(self.A,start_state) + np.dot(self.B,self.U)
        return self.state_pred
    def pred_process_covariance(self,start_cov):
        self.pred_cov = np.dot(np.dot(self.A,start_cov),np.transpose(self.A))
        return self.pred_cov
    def compute_kalman_gain(self,pro__cov_mat):
        self.Kg = np.dot(pro__cov_mat,np.transpose(self.H)) / (np.dot(self.H,np.dot(pro__cov_mat,np.transpose(self.H))) + self.R )
        return self.Kg
    def new_observation(self,old_obs):
        self.new_obs = np.dot(self.C,old_obs) + self.Z
        return self.new_obs
    def compute_current_state(self,prev_state,old_obs,k):
        new_obs = self.new_observation(old_obs)
        new_state = prev_state + np.dot(k,(new_obs - np.dot(self.H,prev_state)))
        return new_state
    def update_process_cov(self,prev_cov,k):
        r = np.eye(4,dtype=np.float32) - np.dot(k,self.H)
        new_cov = np.dot(r,prev_cov)
        return new_cov
    
    def plot(self):
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(positions[:, 0], positions[:, 1], 'b', label='Estimated Position')
        plt.scatter(self.X, self.Y, color='r', marker='x', label='Measurements')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Position Tracking')
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(velocities[:, 0], velocities[:, 1], 'g', label='Estimated Velocity')
        plt.xlabel('X Velocity')
        plt.ylabel('Y Velocity')
        plt.title('Velocity Tracking')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.show()
    
    def start(self):
        for i in range(len(self.X)):
            state_pred = self.predict_state(start_state=self.start_state)
            state_cov = self.pred_process_covariance(self.start_cov)
            Kg = self.compute_kalman_gain(state_cov)
            old_obs = [[self.X[i]],[self.Y[i]],[self.Vx[i]],[self.Vy[i]]]
            new_obs = self.new_observation(old_obs)
            curr_state = np.round(self.compute_current_state(state_pred,new_obs,Kg),2)
            update_cov = self.update_process_cov(prev_cov=state_cov,k=Kg)
            self.start_state = curr_state
            self.start_cov = update_cov
            self.positions.append((curr_state[0, 0], curr_state[1, 0]))
            self.velocities.append((curr_state[2, 0], curr_state[3, 0]))
            
        self.plot()
            
            
tracker = ObjectTracker()
tracker.start()