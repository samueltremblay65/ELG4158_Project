import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from StateCalculation import next_state
from InputGenerator import generateSinusoidAngularInput, generateCircleInput, generateRampInput
from InputGenerator import generateHeartTrajectory
from ErrorCalculation import calculate_average_error

from StateCalculation import simulateSensorData, next_state
from Plotting import plot_results, plot_trajectory, printTrajectoryPath

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints

initial_state, input = generateSinusoidAngularInput(0.2, 0.1, 1, 100)

# Set state to initial state
state = np.transpose(np.array(initial_state))

odometer_uncertainty = [0.05, 0.05, 0.05]
sonar_uncertainty = [0.5,0.5,0.7]

odometer_noise_variance = [0.1, 0.1, 0.025]
sonar_noise_variance = [0.5, 0.5, 0.05]

# Calculate actual states from inputs. Simulate odometer and sonar data
odometer, sonar, actual = simulateSensorData(state, input, odometer_noise_variance, sonar_noise_variance)

Q = np.diag(odometer_uncertainty)
R = np.diag(sonar_uncertainty)

kalman_output = []

# Set initial value of P
P_initial = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

i = 0

def f(state, dt):
    global i
    prediction = next_state(state, input[i-1])
    return prediction

def h(state):
    C = np.array(np.identity(3))
    return C.dot(state)

dt = 1

points = JulierSigmaPoints(3, 0)

unscented_kalman_filter = UnscentedKalmanFilter(3, 3, dt, h, f, points)

unscented_kalman_filter.R = R
unscented_kalman_filter.Q = Q

for idx, sonar_data in enumerate(sonar):
    unscented_kalman_filter.predict()
    unscented_kalman_filter.update(sonar_data)
    kalman_output.append(unscented_kalman_filter.x)
    i = i + 1

print("Odometer only: ")
calculate_average_error(odometer, actual)

print("Sonar only")
calculate_average_error(sonar, actual)

print("UFK Filtered")
calculate_average_error(kalman_output, actual)

plot_results(odometer, sonar, actual, kalman_output)
