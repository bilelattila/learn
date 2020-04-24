import sklearn
import numpy as np
import pandas as pd
import sklearn.tree
from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split
iris=load_iris()

X_train, X_test, Y_train, Y_test=train_test_split(iris.data, iris.target, test_size=0.3)
print(X_train,Y_train)
clf=sklearn.tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
clf.predict([0.1,0.3,0.3,0.4])
