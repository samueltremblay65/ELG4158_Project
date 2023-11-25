import numpy as np

from StateCalculation import next_state

class ExtendedKalmanFilter:
    def __init__(self, state, input, H, Q, R, P_initial):
        self.state = state
        self.input = input
        self.output = None
        self.A = None
        self.C = None
        self.P = P_initial
        self.H = H
        self.Q = Q
        self.R = R

    def set_kalman_matrices(self, A, C):
        self.A = A
        self.C = C
    
    def predict(self, t, noise):
        self.state = next_state(self.state, self.input[t-1]) + noise
        return self.state

    def estimate(self, measurement):
        self.P = self.A @ self.P @ self.A.T + (self.Q)
        measurement_residual = measurement - (self.H @ self.state)
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.state = self.state + (K @ measurement_residual)
        self.P = self.P - (K @ self.H @ self.P)

        self.output = self.C.dot(self.state)

        return self.state, self.output
     