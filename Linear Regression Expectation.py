from sklearn.linear_model import LinearRegression
import numpy as np
import csv

train_input = list()
train_output = list()

filename = 'Winning Percentage vs Run Differential.csv'

with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] == 'data':
            train_input.append([int(row[2]), int(row[3])])
            train_output.append(float(row[1]))

predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=train_input, y=train_output)

x_test = [[693, 784]]
outcome = predictor.predict(X=x_test)
print(outcome)
