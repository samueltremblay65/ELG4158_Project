import numpy as np
from math import sin, cos, acos, asin

def simulateSensorData(state, input, odometer_noise_variance, sonar_noise_variance):
    odometer = [state]
    sonar = [state]
    real = [state]
    input_trajectory = [state]
    odometer_noise_matrix = []

    # Set initial odometer only state to original state
    odometer_state = state
    input_state = state

    for i in range(len(input)):
        process_noise = np.random.normal([0,0,0], [0,0,0], 3)

        input_state = next_state(input_state, input[i])
        state = next_state(state + process_noise, input[i]) # perfect state update

        odometer_noise = np.random.normal([0,0,0], odometer_noise_variance, 3) # Sensor noise from odometer

        odometer_noise_matrix.append(odometer_noise)

        odometer_state = next_state(odometer_state + odometer_noise, input[i]) # Calculate next state using odometer data only

        input_trajectory.append(input_state)
        real.append(state)
        odometer.append(odometer_state)
        sonar.append(state + np.random.normal([0,0,0], sonar_noise_variance, 3))

    return odometer, sonar, real, input_trajectory, odometer_noise_matrix

def next_state(previous_state, input):
    x_next = previous_state[0] + input[0] * cos(previous_state[2])
    y_next = previous_state[1] + input[0] * sin(previous_state[2])
    theta_next = previous_state[2] + input[1]

    return np.transpose(np.array([x_next, y_next, theta_next]))