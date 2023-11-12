import matplotlib.pyplot as plt
import numpy as np

def plot_results(odometer, sonar, perfect_world, kal_out):
    plt.figure(1)
    plt.plot(np.array(perfect_world)[:,0], np.array(perfect_world)[:,1], label='Actual path')
    plt.plot(np.array(sonar)[:,0], np.array(sonar)[:,1], label='Sonar only')
    plt.plot(np.array(odometer)[:,0], np.array(odometer)[:,1], label='Odometer only')
    plt.plot(np.array(kal_out)[:,0], np.array(kal_out)[:,1], label='Filtered signal')
    plt.legend()
    plt.show()