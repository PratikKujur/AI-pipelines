import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        m, n = X.shape
        best_error = float("inf")

        for feature in range(n):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask

                left_value = (
                    np.mean(residuals[left_mask]) if np.sum(left_mask) > 0 else 0
                )
                right_value = (
                    np.mean(residuals[right_mask]) if np.sum(right_mask) > 0 else 0
                )

                pred = np.where(left_mask, left_value, right_value)
                error = np.mean((residuals - pred) ** 2)

                if error < best_error:
                    self.feature_idx = feature
                    best_error = error
                    self.threshold = t
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        mask = X[:, self.feature_idx] <= self.threshold
        return np.where(mask, self.left_value, self.right_value)


class GbmClassifier:
    def __init__(self, n_estimators=20, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.trees = []

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def log_odds(self, y):
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        self.F0 = self.log_odds(y)
        F = np.full(len(y), self.F0)

        for _ in range(self.n_estimators):
            p = self.sigmoid(F)
            residual = y - p

            stump = DecisionStump()
            stump.fit(X, residual)
            update = stump.predict(X)

            F += self.lr * update

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.F0)
        for stump in self.trees:
            F += self.lr * stump.predict(X)
        return self.sigmoid(F)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    clf = GbmClassifier(n_estimators=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)
