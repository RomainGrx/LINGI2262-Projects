#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 16, 18:31:52
@last modified : 2021 Mar 16, 20:24:01
"""

from os.path import join
import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


datasets = join(".", "College")
CollegeTrain = pd.read_csv(join(datasets, "CollegeTrain.csv"), index_col=0)
CollegeTest = pd.read_csv(join(datasets, "CollegeTest.csv"), index_col=0)


# --- Question 1 ---


def get_X_y(df):
    return df.iloc[:, :-1], df.iloc[:, -1]


# get 5% training set
X_train5, y_train5 = get_X_y(CollegeTrain.sample(frac=0.05, random_state=0))
X_test100, y_test100 = get_X_y(CollegeTest.sample(n=100, random_state=0))

# train the decision tree over the 5% train set
clf = DecisionTreeClassifier(random_state=0).fit(X_train5, y_train5)
y_pred100 = clf.predict(X_test100)
test_acc = accuracy_score(y_test100, y_pred100)
print(f"[Q1] Accuracy score for 5% training with 100 test samples :: {test_acc:.3f}")

# --- Question 2 ---

n = 100
alpha = 0.05
z_n = norm.ppf(1 - alpha / 2)
sigma = np.sqrt(test_acc * (1 - test_acc) / n)
lower = test_acc - z_n * sigma
upper = test_acc + z_n * sigma
print(
    f"[Q2] 95% confidence interval for this decision tree :: {lower:.3f}, {upper:.3f}"
)


# --- Question 3 ---

test_accs = []
mean_test_acc = 0.0
for i in range(100):
    X_test100, y_test100 = get_X_y(CollegeTest.sample(n=100, random_state=i))
    acc = accuracy_score(y_test100, clf.predict(X_test100))
    test_accs.append(acc)
    mean_test_acc += acc / 100

# --- Question 4 ---

lower, upper = np.percentile(test_accs, (2.5, 97.5))
print(f"[Q4] observed 95% bounds :: {lower:.3f}, {upper:.3f}")

# --- Question 5 ---


cols = [
    "indiv_test_acc",
    "CI_lower_bound",
    "CI_upper_bound",
    "mean_test_acc",
    "observed_lower_bound",
    "observed_upper_bound",
]

data = []

for i in range(20):
    # get 5% training set
    X_train5, y_train5 = get_X_y(CollegeTrain.sample(frac=0.05, random_state=i))
    X_test100, y_test100 = get_X_y(CollegeTest.sample(n=100, random_state=i))

    # train the decision tree over the 5% train set
    clf = DecisionTreeClassifier(random_state=0).fit(X_train5, y_train5)
    y_pred100 = clf.predict(X_test100)
    test_acc = accuracy_score(y_test100, y_pred100)

    # compute the theorical bounds
    n = 100
    alpha = 0.05
    z_n = norm.ppf(1 - alpha / 2)
    sigma = np.sqrt(test_acc * (1 - test_acc) / n)
    lower = test_acc - z_n * sigma
    upper = test_acc + z_n * sigma

    # test on 100 distincts samples
    test_accs = []
    mean_test_acc = 0.0
    for j in range(100):
        X_test100, y_test100 = get_X_y(CollegeTest.sample(n=100, random_state=i * j))
        acc = accuracy_score(y_test100, clf.predict(X_test100))
        test_accs.append(acc)
        mean_test_acc += acc / 100

    # get the observed bounds
    observed_lower, observed_upper = np.percentile(test_accs, (2.5, 97.5))

    data.append(
        [
            test_acc,
            lower,
            upper,
            mean_test_acc,
            observed_lower,
            observed_upper,
        ]
    )

frame = pd.DataFrame(data, columns=cols)












