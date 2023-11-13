import matplotlib.pyplot as plt
import numpy as np

def plot_results(odometer, sonar, actual, kalman_out):
    plt.figure(1)
    plt.plot(np.array(actual)[:,0], np.array(actual)[:,1], label='Actual path')
    plt.plot(np.array(sonar)[:,0], np.array(sonar)[:,1], label='Sonar only')
    plt.plot(np.array(odometer)[:,0], np.array(odometer)[:,1], label='Odometer only')
    plt.plot(np.array(kalman_out)[:,0], np.array(kalman_out)[:,1], label='Filtered signal')
    plt.legend()
    plt.show()

def plot_trajectory(trajectory):
    plt.figure(1)
    plt.plot(np.array(trajectory)[:,0], np.array(trajectory)[:,1], label='Actual path')
    plt.legend()
    plt.show()

def printTrajectoryPath(trajectory):
    for i in range(len(trajectory)):
        print(str(i) + ": " + str(trajectory[i]) +" rad")
