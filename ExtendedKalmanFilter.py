import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from InputGenerator import generateSinusoidAngularInput, generateCircleInput, generateRampInput
from InputGenerator import generateHeartTrajectory

from StateCalculation import simulateSensorData, next_state
from Plotting import plot_results, plot_trajectory, printTrajectoryPath

class ExtendedKalmanFilter:
    def __init__(self, state, input, H, Q, R, P_initial):
        self.state = state
        self.input = input
        self.output = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.P = P_initial
        self.H = H
        self.Q = Q
        self.R = R

    def set_kalman_matrices(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    
    def predict(self, t):
        self.state = next_state(self.state, self.input[t-1])
        return self.state

    def estimate(self, measurement):
        self.P = self.A @ self.P @ self.A.T + (self.Q)
        # print(self.P)
        measurement_residual = measurement - (self.H @ self.state)
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.state = self.state + (K @ measurement_residual)
        self.P = self.P - (K @ self.H @ self.P)

        self.output = self.C.dot(self.state)

        return self.state, self.output
     
def main():
    np.set_printoptions(precision=3,suppress=True)

    # process_noise = np.array([0.01, 0.01, 0.003])

    initial_state, input = generateSinusoidAngularInput(0.2, 0.1, 1, 100)

    # Set state to initial state
    state = np.transpose(np.array(initial_state))

    odometer_uncertainty = [0.05, 0.05, 0.05]
    sonar_uncertainty = [0.5,0.5,0.7]

    odometer_noise_variance = [0.05, 0.05, 0.025]
    sonar_noise_variance = [0.3, 0.3, 0.05]

    # Calculate actual states from inputs. Simulate odometer and sonar data
    odometer, sonar, actual = simulateSensorData(state, input, odometer_noise_variance, sonar_noise_variance)

    A = np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    Q = np.diag(odometer_uncertainty)
    R = np.diag(sonar_uncertainty)         

    kalman_output = [initial_state]

    # Set initial value of P
    P_initial = np.array([[0.185, 0, 0], [0, 0.185, 0], [0, 0, 0.214]])
    extended_kalman_filter = ExtendedKalmanFilter(state, input, H, Q, R, P_initial)

    for idx, sonar_data in enumerate(sonar):
        A = np.array(np.identity(3))
        B = np.array([[cos(state[2]), 0], [sin(state[2]), 0], [0,1]])
        C = np.array(np.identity(3))
        D = np.zeros((2,2))

        extended_kalman_filter.set_kalman_matrices(A,B,C,D)
        extended_kalman_filter.predict(idx)
        state, output = extended_kalman_filter.estimate(sonar_data)
        kalman_output.append(output)

    plot_results(odometer, sonar, actual, kalman_output)
 
main()