import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Decision Tree")
    index = 512
    data, target = load_digits(return_X_y=True)
    # print(target[index])
    # plt.imshow(data[index].reshape(8,8), cmap='gray')
    # plt.show()
    clf = DecisionTreeClassifier(criterion="entropy")
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.1, random_state=40)
    clf.fit(train_X, train_y)
    predict = clf.predict(test_X)
    predict_train = clf.predict(train_X)
    print("accuracy\t","test: ", accuracy_score(test_y, predict),"\ttrain", accuracy_score(train_y, predict_train))

    # plot_tree(clf)
    # plt.show()
