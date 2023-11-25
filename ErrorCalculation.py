from math import sqrt

def calculate_average_error(actual, reference):
    if len(actual) != len(reference):
        print("Arrays are incompatible")

    errors = []
    
    for idx in range(len(actual)):
        errors.append([abs(actual[idx][0] - reference[idx][0]), abs(actual[idx][1] - reference[idx][1]), abs(actual[idx][2] - reference[idx][2])])
    
    average_x_error = 0
    average_y_error = 0
    average_theta_error = 0

    for value in errors:
        average_x_error += value[0]
        average_y_error += value[1]
        average_theta_error += value[2]

    average_x_error = average_x_error / (len(actual))
    average_y_error = average_y_error / (len(actual))
    average_theta_error = average_theta_error / (len(actual))

    return average_x_error, average_y_error, average_theta_error


def calculate_root_mean_square(actual, reference):
    if len(actual) != len(reference):
        print("Arrays are incompatible")

    errors = []
    
    for idx in range(len(actual)):
        error = [actual[idx][0] - reference[idx][0], actual[idx][1] - reference[idx][1], actual[idx][1] - reference[idx][1]]
        error[0] = error[0] * error[0]
        error[1] = error[1] * error[1]
        error[2] = error[2] * error[2]
        errors.append(error)
    
    average_x_error = 0
    average_y_error = 0
    average_theta_error = 0

    for value in errors:
        average_x_error += value[0]
        average_y_error += value[1]
        average_theta_error += value[2]

    average_x_error = average_x_error / (len(actual))
    average_y_error = average_y_error / (len(actual))
    average_theta_error = average_theta_error / (len(actual))

    return sqrt(average_x_error), sqrt(average_y_error), sqrt(average_theta_error)

def printErrorVector(title, vector):
    print(title)
    print("x error: " + str(vector[0]))
    print("y error: " + str(vector[1]))
    print("theta error: " + str(vector[2]))
    print("")
