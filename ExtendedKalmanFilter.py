import numpy as np
import matplotlib as plt

np.set_printoptions(precision=3,suppress=True)
 
A = np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
 
# Noise applied to the forward kinematics (calculation
# of the estimated state at time k from the state
# transition model of the mobile robot). This is a vector
# with the number of elements equal to the number of states
process_noise = np.array([0.01, 0.01, 0.003])
     
# State model noise covariance matrix Q
# When Q is large, the Kalman Filter tracks large changes in 
# the sensor measurements more closely than for smaller Q.
# Q is a square matrix that has the same number of rows as states.
Q = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
                 
# Measurement matrix H
# Used to convert the predicted state estimate at time k
# into predicted sensor measurements at time k.
# In this case, H will be the identity matrix since the 
# estimated state maps directly to state measurements from the 
# odometry data [x, y, theta]
# H has the same number of rows as sensor measurements
# and same number of columns as states.
H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
                         
# Sensor measurement noise covariance matrix R
# Has the same number of rows and columns as sensor measurements.
# If we are sure about the measurements, R will be near zero.
R = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])  
                 
# Sensor noise. This is a vector with the
# number of elements equal to the number of sensor measurements.
sensor_noise = np.array([0.07, 0.07, 0.04])

def getB(theta, delta_k):
    """
    Calculates and returns the B matrix
    3x2 matix -> number of states x number of control inputs
    The control inputs are the forward speed and the
    rotation rate around the z axis from the x-axis in the 
    counterclockwise direction.
    [v,theta_rate]
    Expresses how the state of the system [x,y,theta] changes
    from k-1 to k due to the control commands (i.e. control input).
    :param theta: The theta angle (rotation angle around the z axis) in rad 
    :param delta_k: The change in time from time step k-1 to k in sec
    """
    B = np.array([[np.cos(theta)*delta_k, 0], [np.sin(theta)*delta_k, 0], [0, delta_k]])
    return B
 
def ekf(z_k_observation_vector, state_estimate, 
        input, P, dk):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to 
    create an optimal estimate of the state of the robotic system.
         
    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,theta] in the global reference frame
            in [meters,meters,radians].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,theta] in the global reference frame
            in [meters,meters,radians].
        :param input The control vector applied at time k-1
            3x1 NumPy Array [v,v,theta rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds
             
    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            3x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array                 
    """
    ######################### Predict #############################
    # Predict the state estimate at time k based on the state 
    # estimate at time k-1 and the control input applied at time k-1.
    state_estimate_k = A @ (state_estimate) + (getB(state_estimate[2], dk)) @ (input) + (process_noise)
             
    print(f'State Estimate Before EKF={state_estimate_k}')
             
    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    P_k = A @ P @ A.T + (Q)
         
    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted 
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector - (
            (H @ state_estimate_k) + (
            sensor_noise))
 
    print(f'Observation={z_k_observation_vector}')
             
    # Calculate the measurement residual covariance
    S_k = H @ P_k @ H.T + R_k
         
    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be
    # non-square or singular.
    K_k = P_k @ H.T @ np.linalg.pinv(S_k)
         
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
     
    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H @ P_k)
     
    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={state_estimate_k}')
 
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
     
def main():
 
    # We start at time k=1
    k = 1
     
    # Time interval in seconds
    dk = 1
 
    # Create a list of sensor observations at successive timesteps
    # Each list within z_k is an observation vector.
    z_k = np.array([[4.721,0.143,0.006], # k=1
                    [9.353,0.284,0.007], # k=2
                    [14.773,0.422,0.009],# k=3
                    [18.246,0.555,0.011], # k=4
                    [22.609,0.715,0.012]])# k=5
                     
    # The estimated state vector at time k-1 in the global reference frame.
    # [x_k_minus_1, y_k_minus_1, theta_k_minus_1]
    # [meters, meters, radians]
    state_estimate = np.array([0.0,0.0,0.0])
     
    # The control input vector at time k-1 in the global reference frame.
    # [v, theta_rate]
    # [meters/second, radians/second]
    # In the literature, this is commonly u.
    # Because there is no angular velocity and the robot begins at the 
    # origin with a 0 radians theta angle, this robot is traveling along 
    # the positive x-axis in the global reference frame.
    input = np.array([4.5,0.0])
     
    # State covariance matrix P
    # This matrix has the same number of rows (and columns) as the 
    # number of states (i.e. 3x3 matrix). P is sometimes referred
    # to as Sigma in the literature. It represents an estimate of 
    # the accuracy of the state estimate at time k made using the
    # state transition matrix. We start off with guessed values.
    P = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
                             
    # Start at k=1 and go through each of the 5 sensor observations, 
    # one at a time. 
    # We stop right after timestep k=5 (i.e. the last sensor observation)
    for k, observation_vector in enumerate(z_k,start=1):
     
        # Print the current timestep
        print(f'Timestep k={k}')  
         
        # Run the Extended Kalman Filter and store the 
        # near-optimal state and covariance estimates
        optimal_state_estimate, covariance_estimate_k = ekf(
            observation_vector, # Most recent sensor measurement
            state_estimate, # Our most recent estimate of the state
            input, # Our most recent control input
            P, # Our most recent state covariance matrix
            dk) # Time interval
         
        # Get ready for the next timestep by updating the variable values
        state_estimate = optimal_state_estimate
        P = covariance_estimate_k
         
        # Print a blank line
        print()
 
# Program starts running here with the main method  
main()