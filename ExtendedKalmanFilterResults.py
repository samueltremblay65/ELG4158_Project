import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from InputGenerator import generateSinusoidAngularInput, generateRampInput

from StateCalculation import simulateSensorData, next_state
from Plotting import plot_results, plot_trajectory, printTrajectoryPath
from ErrorCalculation import calculate_average_error
from ExtendedKalmanFilter import ExtendedKalmanFilter

np.set_printoptions(precision=3,suppress=True)

# process_noise = np.array([0.01, 0.01, 0.003])

initial_state, input = generateSinusoidAngularInput(0.2, 0.1, 1, 100)

# Set state to initial state
state = np.transpose(np.array(initial_state))

odometer_uncertainty = [0.05, 0.05, 0.05]
sonar_uncertainty = [0.5,0.5,0.7]

odometer_noise_variance = [0.1, 0.1, 0.05]
sonar_noise_variance = [0.3, 0.3, 0.05]

# Calculate actual states from inputs. Simulate odometer and sonar data
odometer, sonar, real, input_trajectory, odometer_noise = simulateSensorData(state, input, odometer_noise_variance, sonar_noise_variance)

A = np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

Q = np.diag(odometer_uncertainty)
R = np.diag(sonar_uncertainty)         

kalman_output = []

# Set initial value of P
P_initial = np.array([[0.185, 0, 0], [0, 0.185, 0], [0, 0, 0.214]])
extended_kalman_filter = ExtendedKalmanFilter(state, input, H, Q, R, P_initial)

for idx, sonar_data in enumerate(sonar):
    A = np.array(np.identity(3))
    C = np.array(np.identity(3))

    extended_kalman_filter.set_kalman_matrices(A,C)
    extended_kalman_filter.predict(idx, odometer_noise[idx-1])
    state, output = extended_kalman_filter.estimate(sonar_data)
    kalman_output.append(output)

print("Odometer only: ")
calculate_average_error(odometer, real)

print("Sonar only")
calculate_average_error(sonar, real)

print("EKF Filtered")
calculate_average_error(kalman_output, real)

plot_results(odometer, sonar, real, kalman_output)