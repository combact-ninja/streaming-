
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import deque

# Step 1: DBN-like model using MLP
class DBNClassifier:
    def __init__(self, hidden_layer_sizes=(100,)):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1, warm_start=True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def partial_fit(self, X, y, classes):
        self.model.partial_fit(X, y, classes=classes)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Step 2: Adaptive Sliding Window Drift Detection
class DriftDetector:
    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.reference_window = deque(maxlen=window_size)
        self.drift_detected = False

    def add_element(self, element):
        self.window.append(element)
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(element)
        else:
            self.detect_drift()

    def detect_drift(self):
        reference_mean = np.mean(self.reference_window)
        current_mean = np.mean(self.window)
        if abs(reference_mean - current_mean) > self.threshold:
            self.drift_detected = True
            self.reference_window.clear()
        else:
            self.drift_detected = False

# Step 3: Integrate Everything
def DBN_with_Drift_Detection(X_train, y_train, X_test, y_test):
    dbn = DBNClassifier(hidden_layer_sizes=(500, 200))
    drift_detector = DriftDetector(window_size=100, threshold=0.1)

    classes = np.unique(y_train)

    # Initial training on the training data
    dbn.fit(X_train, y_train)

    y_pred = []
    y_true = []
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for xi, yi in zip(X_test, y_test):
        xi = xi.reshape(1, -1)

        # Predict and record metrics
        yi_pred = dbn.predict(xi)
        y_pred.append(yi_pred[0])
        y_true.append(yi)

        # Update the drift detector
        drift_detector.add_element(yi_pred == yi)

        # If drift is detected, reset the model
        if drift_detector.drift_detected:
            print("Drift detected. Resetting model.")
            dbn = DBNClassifier(hidden_layer_sizes=(500, 200))
            dbn.fit(X_train, y_train)

        # Continue training the model incrementally
        dbn.partial_fit(xi, yi, classes=classes)

        # Calculate and store metrics
        metrics['accuracy'].append(accuracy_score([yi], yi_pred))
        metrics['precision'].append(precision_score([yi], yi_pred, average='macro'))
        metrics['recall'].append(recall_score([yi], yi_pred, average='macro'))
        metrics['f1'].append(f1_score([yi], yi_pred, average='macro'))

    print("Final Metrics:")
    print("Accuracy:", np.mean(metrics['accuracy']))
    print("Precision:", np.mean(metrics['precision']))
    print("Recall:", np.mean(metrics['recall']))
    print("F1 Score:", np.mean(metrics['f1']))

    return metrics

# Example usage:
# Generate some synthetic data
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 8, 1000)
X_test = np.random.randn(200, 784)
y_test = np.random.randint(0, 8, 200)

# Run the model with drift detection
metrics = DBN_with_Drift_Detection(X_train, y_train, X_test, y_test)
