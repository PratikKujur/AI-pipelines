import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


def sigmoid(z):
    return np.where(z >= 0.5, 1, 0)


class PerceptronPipeline:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.sigmoid = sigmoid

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        for _ in range(self.epochs):
            for i in range(m):
                y_pred = self.sigmoid(z=np.dot(X[i],self.w) + self.b)
                update = y[i] - y_pred
                self.w = self.w + update * self.lr
                self.b = self.b + update * self.lr

    def predict(self, X):
        return sigmoid(z=X@self.w+self.b)


if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = PerceptronPipeline(lr=0.01, epochs=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))
