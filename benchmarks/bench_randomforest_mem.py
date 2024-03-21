from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

n_samples = 10_000
n_features = 200
n_estimators = 200

# Generate synthetic classification dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)

# Initialize RandomForestClassifier
clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

# Fit the classifier on the training data
clf.fit(X, y)
