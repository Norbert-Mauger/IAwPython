import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from utilities import visualize_classifier
from utilities import visualize_dataset


# Define sample input data
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9], [2, 3], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 3])

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

# Train the classifier
classifier.fit(X, y)

# Visualize the performance of the classifier
# visualize_classifier(classifier, X, y)

vals = np.array([2, 6, 3, 3, 4, 8, -6, 1, 2, 5, 6, 9, 7, 34, 7, 12, 1, 14, 8])

visualize_dataset(vals, 'ploum')

