import numpy as np
from math import sin, cos

def generateSinusoidAngularInput(a, angular_frequency, linear_velocity, length):
    inputSequence = np.zeros((length, 2))
    for i in range(length):
        inputSequence[i][0] = linear_velocity
        inputSequence[i][1] = a * sin(float(i) * angular_frequency)

    return inputSequence

def generateCircleInput(angular_rate, linear_velocity, length):
    inputSequence = np.zeros((length, 2))
    for i in range(length):
        inputSequence[i][0] = linear_velocity
        inputSequence[i][1] = angular_rate

    return inputSequence

def generateRampInput(linear_velocity, length):
    inputSequence = np.zeros((length, 2))
    for i in range(length):
        inputSequence[i][0] = linear_velocity
        inputSequence[i][1] = 0

    return inputSequence