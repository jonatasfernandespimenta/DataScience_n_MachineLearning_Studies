from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import tkinter

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

matplotlib.use(backend="TkAgg")

mnist = fetch_openml('mnist_784', version=1, cache=False)
mnist.target = mnist.target.astype(np.int8)

X, y = mnist["data"], mnist["target"]

some_digit = X[777]

# print(y[777])

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis('off')
# plt.show()

# print(y[777])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Treinando um Binary Classifier
y_train_8 = (y_train == 8)
y_test_8 = (y_test == 8)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_8)
print(sgd_clf.predict([some_digit]))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_8, y_scores)


def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


# plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
# plt.show()

# Roc Curve

fpr, tpr, thresholds = roc_curve(y_train_8, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# plot_roc_curve(fpr, tpr)
# plt.show()

# roc_auc_score(y_train_8, y_scores)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
sgd_clf = SGDClassifier(random_state=42)
print(0)
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
print(1)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(2)
# The matrix returns a lot of number so it's better if we show it as an image
plt.matshow(conf_mx, cmap=plt.cm.gray)
print(3)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
