from sklearn.linear_model import LinearRegression
import csv
import os

train_input = list()
train_output = list()

for file in os.listdir('Testing Data'):
    if file.endswith('.csv'):
        rout = 'Testing Data/' + file
        print(rout)
        with open(rout, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'data':
                    train_input.append([int(row[2]), int(row[3])])
                    train_output.append(float(row[1]))

predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=train_input, y=train_output)

x_test = [[693, 784]]  # Perfect prediction would yield .469, pythagorean expectation is .439
outcome = predictor.predict(X=x_test)
print(outcome)
