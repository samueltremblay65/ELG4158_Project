
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

    print("Average x error: " + str(average_x_error))
    print("Average y error: " + str(average_y_error))
    print("Average theta error: " + str(average_theta_error))

        


