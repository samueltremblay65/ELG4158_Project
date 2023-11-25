import matplotlib.pyplot as plt
import numpy as np

def plot_results(odometer, sonar, actual, kalman_out):
    plt.plot(np.array(actual)[:,0], np.array(actual)[:,1], label='Actual path')
    plt.plot(np.array(sonar)[:,0], np.array(sonar)[:,1], label='Sonar only')
    plt.plot(np.array(odometer)[:,0], np.array(odometer)[:,1], label='Odometer only')
    plt.plot(np.array(kalman_out)[:,0], np.array(kalman_out)[:,1], label='Filtered signal')
    plt.legend()
    plt.show()

def plot_filter_comparison(actual, ekf, ukf):
    plt.plot(np.array(actual)[:,0], np.array(actual)[:,1], label='Actual path')
    plt.plot(np.array(ekf)[:,0], np.array(ekf)[:,1], label='Extended Kalman Filter')
    plt.plot(np.array(ukf)[:,0], np.array(ukf)[:,1], label='Unscented Kalman Filter')
    plt.legend()
    plt.show()

def plot_trajectory(trajectory):
    plt.plot(np.array(trajectory)[:,0], np.array(trajectory)[:,1], label='Actual path')
    plt.legend()
    plt.show()

def printTrajectoryPath(trajectory):
    for i in range(len(trajectory)):
        print(str(i) + ": " + str(trajectory[i]) +" rad")
