"""
    This Code implements tracking of the horizontal and vertical distance of a projectile in 2D (x,y) and the velocities in x and y.
    It assumes angle of projection to be 90 degrees.
"""
import numpy as np

class ObjectTracker():
    def __init__(self,initial_pos):
        self.prev_pos = initial_pos
        self.start_cov = np.array([[1e-3, 0, 0, 0],[0, 1e-3, 0, 0],[0, 0, 1e-2, 0],[0, 0, 0, 1e-2]])
        self.dt = 1
        self.g = 9.8
        self.B = np.array([[0.5*self.dt**2,0],[0,0.5*self.dt**2],[self.dt,0],[0,self.dt]],dtype=np.float32)
        self.U = np.array([[0],[self.g]],dtype=np.float32)
        self.A = np.eye(4,dtype=np.float32)
        self.A[0][2] = self.dt
        self.A[1][3] = self.dt
        
        
    def predict_state(self):
        self.state_pred = np.dot(self.A,self.prev_pos) + np.dot(self.B,self.U)
        return self.state_pred
    def pred_process_covariance(self,prev_cov):
        self.pred_cov = np.dot(np.dot(self.A,prev_cov),np.transpose(self.A))
        return self.pred_cov
    def compute_kalman_gain(self):
        






initial_pos = np.array([[0],[0],[0],[0]])
tracker = ObjectTracker(initial_pos=initial_pos)
print(tracker.predict_state())
print(tracker.pred_process_covariance(tracker.start_cov))