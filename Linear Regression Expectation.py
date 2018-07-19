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
                print('My program is more accurate')
            elif abs(float(py_expectation) - float(row[1])) < abs(float(outcome) - float(row[1])):
                print('Pythagorean expectation is more accurate')
            else:
                print('This is highly unlikely')

# x_test = [[693, 784]]  # Perfect prediction would yield .469, pythagorean expectation is .439
# outcome = predictor.predict(X=x_test)
# print(outcome)
