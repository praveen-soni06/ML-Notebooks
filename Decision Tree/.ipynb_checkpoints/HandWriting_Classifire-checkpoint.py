import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\DataScience\Machine Learning Algorithms\Decision Tree\train.csv")
data.shape
data.info()
X = data.iloc[:,1:].values
print(X)
y = data.iloc[:,0].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=9)
from sklearn.tree import DecisionTreeClassifier
D_clf = DecisionTreeClassifier()
D_clf.fit(X_train, y_train)
# check what is store in 101 position in y_test

y_test[101]
# lets print the fig

plt.imshow(X_test[102].reshape(28,28))
print("pridicted number is: " ,D_clf.predict(X_test[102].reshape(1,784)))
# try to print all the numbers
for i in range(0,200,10):
    plt.figure(figsize=(10,5))
    
    for j in range(10):
        plt.subplot(2,5,j+1)
        plt.imshow(X_test[i+j].reshape(28,28))
        plt.title(f"Title: {y_test[i+j]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
%history -f HandWriting_Classifire.py
