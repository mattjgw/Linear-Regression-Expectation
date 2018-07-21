from sklearn.linear_model import LinearRegression
import csv
import os


def pythagorean_expectation(runs_scored, runs_allowed):
    return runs_scored**2 / (runs_scored**2 + runs_allowed**2)


train_input = list()
train_output = list()

for file in os.listdir('Testing Data'):
    if file.endswith('.csv'):
        rout = 'Testing Data/' + file
        with open(rout, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row[0] == 'data':
                    train_input.append([int(row[2]), int(row[3])])
                    train_output.append(float(row[1]))

predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=train_input, y=train_output)

pythagorean_results = list()
linear_regression_results = list()
actual_results = list()

with open('2017 Winning Percentage vs Run Differential.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if row[0] == 'data':
            print()
            py_expectation = pythagorean_expectation(int(row[2]), int(row[3]))
            x = [[int(row[2]), int(row[3])]]
            outcome = predictor.predict(X=x)

            print('The winning percentage is: ' + str(row[1]))
            print('The pythagorean expectation is: ' + str(py_expectation))
            print('The model predicted: ' + str(outcome))

            if abs(float(py_expectation) - float(row[1])) > abs(float(outcome) - float(row[1])):
                print('In this case, the linear regression model is more accurate')
            elif abs(float(py_expectation) - float(row[1])) < abs(float(outcome) - float(row[1])):
                print('In this case, the pythagorean expectation is more accurate')
            else:
                print('This is highly unlikely')

            pythagorean_results.append(float(py_expectation))
            linear_regression_results.append(float(outcome))
            actual_results.append(float(row[1]))

pythagorean_error = list()
linear_regression_error = list()

for i in range(len(actual_results)):
    pythagorean_error.append(abs(actual_results[i] - pythagorean_results[i]))
    linear_regression_error.append(abs(actual_results[i] - linear_regression_results[i]))

pythagorean_average_error = sum(pythagorean_error) / len(pythagorean_error)
linear_regression_average_error = sum(linear_regression_error) / len(linear_regression_error)

print()
print('The average error in the pythagorean expectation is ' + str(pythagorean_average_error*100) + ' percent')
print('The average error in the linear regression prediction is ' + str(linear_regression_average_error*100)
      + ' percent')

print('The linear regression prediction is ' + str(100 - (linear_regression_average_error/pythagorean_average_error*100))
      + ' percent more accurate')
