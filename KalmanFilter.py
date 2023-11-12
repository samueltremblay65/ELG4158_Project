import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from InputGenerator import generateSinusoidAngularInput, generateCircleInput, generateRampInput

from StateCalculation import simulateSensorData, next_state
from Plotting import plot_results

class UnscentedKalmanFilter:
    def __init__(self, state, input):
        self.state = state
        self.input = input
        self.output = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.sigma_state = None
        self.sigma_output = None
        self.P = None
    
    def set_kalman_matrices(self, A, B, C, D, P, sigma_state, sigma_output):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.sigma_state = sigma_state
        self.sigma_output = sigma_output
        self.P = P
    
    def predict(self, t):
        self.state = next_state(self.state, self.input[t]) + np.random.normal([0,0,0], [0.1, 0.1, 0.001], 3)
        self.P = self.A*self.P*np.transpose(A) + self.sigma_state
        self.output = C.dot(self.state)
        return self.state

    def estimate(self, measurement):
        kalman_gain = self.P*np.transpose(C) * np.linalg.pinv((C*self.P*np.transpose(C) + self.sigma_output))
        gain_factor = kalman_gain.dot((measurement - self.output))
        self.state = self.state + gain_factor
        self.P = (np.diag([1,1,1]) - kalman_gain*self.C)*self.P
        self.output = C.dot(self.state)

        return self.state, self.output 

if __name__ == "__main__":

    simulation_length = 100

    # Input sequence
    # input = generateSinusoidAngularInput(0.1, 0.1, 1, simulation_length)
    input = generateRampInput(1, simulation_length)

    # Initial state
    state = np.transpose(np.array([0,0,0]))
    
    # Covariance matrices
    sigma_x = 0.05
    sigma_y = 0.05
    sigma_theta = 0.25

    odometer_uncertainty = [0.1, 0.1, 0.1]
    sonar_uncertainty = [0.5,0.5,0.5]

    # Tuning parameters
    P = np.diag([sigma_x ** 2, sigma_y ** 2, sigma_theta ** 2])
    sigma_state = np.diag(odometer_uncertainty) # Uncertainity in odometer prediction
    sigma_output = np.diag(sonar_uncertainty) # Uncertainity in sonar measurement

    odometer_noise_variance = [0.05, 0.05, 0.01]
    sonar_noise_variance = [0.5, 0.5, 0.1]

    # Calculate actual states from inputs. Simulate odometer and sonar data
    odometer, sonar, actual = simulateSensorData(state, input, odometer_noise_variance, sonar_noise_variance)

    kalman_output = []

    unscented_kalman_filter = UnscentedKalmanFilter(state, input)

    for idx, sonar_data in enumerate(sonar):
        A = np.array(np.identity(3))
        B = np.array([[cos(state[2]), 0], [sin(state[2]), 0], [0,1]])
        C = np.array(np.identity(3))
        C[2][2] = 0
        D = np.zeros((2,2))

        unscented_kalman_filter.set_kalman_matrices(A,B,C,D,P,sigma_state, sigma_output)
        unscented_kalman_filter.predict(idx)
        state, output = unscented_kalman_filter.estimate(sonar_data)
        kalman_output.append(output)

    plot_results(odometer, sonar, actual, kalman_output)