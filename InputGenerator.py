import numpy as np
from math import sin, cos, pi

def generateSinusoidAngularInput(a, angular_frequency, linear_velocity, length):
    inputSequence = np.zeros((length, 2))
    for i in range(length):
        inputSequence[i][0] = linear_velocity
        inputSequence[i][1] = a * sin(float(i) * angular_frequency)

    return [0,0,0], inputSequence

def generateCircleInput(angular_rate, linear_velocity, length):
    inputSequence = np.zeros((length, 2))
    for i in range(length):
        inputSequence[i][0] = linear_velocity
        inputSequence[i][1] = angular_rate

    return [0,0,0] , inputSequence

def generateRampInput(linear_velocity, length):
    inputSequence = np.zeros((length, 2))
    for i in range(length):
        inputSequence[i][0] = linear_velocity
        inputSequence[i][1] = 0

    return [0,0,0], inputSequence

def generateHeartTrajectory():
    initial_state = [0,0, 53 * pi / 180]
    inputSequence = np.zeros((47, 2))

    turn_sequence = np.zeros(23)

    for i in range(23):
        turn_sequence[i] = 0.001 * i * i

    for i in range(23):
        inputSequence[i][0] = 1
        inputSequence[i][1] = turn_sequence[i]

    inputSequence[22][0] = 1.753
    inputSequence[22][1] = 0

    inputSequence[23][0] = 0
    inputSequence[23][1] = -2.188899693

    for i in range(24, 47):
        inputSequence[i][0] = 1
        inputSequence[i][1] = turn_sequence[22 - i + 23]

    inputSequence[24][0] = 1.753

    return initial_state, inputSequence

    