import numpy as np
from math import sin, cos

def simulateSensorData(state, input, odometer_noise_variance, sonar_noise_variance):
    odometer = []
    sonar = []
    actual = []

    # Set initial odometer only state to original state
    odometer_state = state

    for i in range(len(input)):
        state = next_state(state, input[i]) # perfect state update

        odometer_noise = np.random.normal([0,0,0], odometer_noise_variance, 3) # Sensor noise from odometer

        odometer_state = next_state(odometer_state + odometer_noise, input[i]) # Calculate next state using odometer data only

        actual.append(state)
        odometer.append(odometer_state)
        sonar.append(state + np.random.normal([0,0,0], sonar_noise_variance, 3))

    return odometer, sonar, actual

def next_state(previous_state, input):
    x_next = previous_state[0] + input[0] * cos(previous_state[2])
    y_next = previous_state[1] + input[0] * sin(previous_state[2])
    theta_next = previous_state[2] + input[1]

    return np.transpose(np.array([x_next, y_next, theta_next]))