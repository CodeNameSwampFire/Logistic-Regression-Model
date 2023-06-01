import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataset = pd.read_csv('Breast_cancer.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y= y_train, cv = 10)  #cv is for getting the number of folds
acc = format(accuracies.mean()*100, ".2f")
standard_deviation = format(accuracies.std()*100, "0.2f")
print(acc)
print(standard_deviation)


