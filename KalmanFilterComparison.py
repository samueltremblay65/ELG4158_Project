import numpy as np
import matplotlib.pyplot as plt
from StateCalculation import next_state
from InputGenerator import generateSinusoidAngularInput, generateRampInput
from ErrorCalculation import calculate_average_error, calculate_root_mean_square, printErrorVector

from StateCalculation import simulateSensorData, next_state
from Plotting import plot_results, plot_filter_comparison, printTrajectoryPath

from ExtendedKalmanFilter import ExtendedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints

np.set_printoptions(precision=3,suppress=True)

initial_state, input = generateRampInput(1, 100)

odometer_uncertainty = [0.01, 0.01, 0.01]
sonar_uncertainty = [5, 5, 0.1]

odometer_noise_variance = [0.01, 0.01, 0.01]
sonar_noise_variance = [0.5, 0.5, 0.1]

initial_state = np.transpose(np.array(initial_state))

# Calculate actual states from inputs. Simulate odometer and sonar data
odometer, sonar, real, input_trajectory, odometer_noise = simulateSensorData(initial_state, input, odometer_noise_variance, sonar_noise_variance)

Q = np.diag(odometer_uncertainty)
R = np.diag(sonar_uncertainty)

# EKF setup
ekf_state = initial_state
A = np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

ekf_output = []
ukf_output = []

# Set initial value of P
P_initial = np.array([[0.185, 0, 0], [0, 0.185, 0], [0, 0, 0.214]])
extended_kalman_filter = ExtendedKalmanFilter(ekf_state, input, H, Q, R, P_initial)

# UKD setup
def f(state, dt):
    global i
    prediction = next_state(state + odometer_noise[i-1], input[i-1])
    return prediction

def h(state):
    C = np.array(np.identity(3))
    return C.dot(state)

i = 0
dt = 1

ukf_state = initial_state

points = JulierSigmaPoints(3, 0)

unscented_kalman_filter = UnscentedKalmanFilter(3, 3, dt, h, f, points)

unscented_kalman_filter.R = R
unscented_kalman_filter.Q = Q

# Computing ekf_output
for idx, sonar_data in enumerate(sonar):
    A = np.array(np.identity(3))
    C = np.array(np.identity(3))

    extended_kalman_filter.set_kalman_matrices(A,C)
    extended_kalman_filter.predict(idx, odometer_noise[idx-1])
    state, output = extended_kalman_filter.estimate(sonar_data)
    ekf_output.append(output)

# Computing ukf_output
for idx, sonar_data in enumerate(sonar):
    unscented_kalman_filter.predict()
    unscented_kalman_filter.update(sonar_data)
    ukf_output.append(unscented_kalman_filter.x)
    i = i + 1


printErrorVector("Odometer only, average error", calculate_average_error(odometer, real))

printErrorVector("Sonar only, average error", calculate_average_error(sonar, real))

printErrorVector("EKF output, average error", calculate_average_error(ekf_output, real))

printErrorVector("UKF output, average error", calculate_average_error(ukf_output, real))

printErrorVector("Odometer only, RMS error", calculate_average_error(odometer, real))

printErrorVector("Sonar only, RMS error", calculate_average_error(sonar, real))

printErrorVector("EKF output,  RMS error", calculate_average_error(ekf_output, real))

printErrorVector("UKF output,  RMS error", calculate_average_error(ukf_output, real))

plt.figure(1)
plt.title("Comparison of EKF output with sonar and odometer measurements")
plot_results(odometer, sonar, real, ekf_output)
plt.figure(2)
plt.title("Comparison of EKF output with sonar and odometer measurements")
plot_results(odometer, sonar, real, ukf_output)
plt.figure(3)
plot_filter_comparison(real, ekf_output, ukf_output)
plt.title("Comparison of EKF and UKF outputs")


