import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def visualize_dataset(X, title='test'):

    min_x, max_x = 0, len(X)
    min_y, max_y = X.min()-1, X.max()+1

    plt.figure()
    plt.title(title)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # plot data points
    plt.scatter(range(len(X)), X, s=75)

    # plot hz line representing the mean value
    plt.hlines(X.mean(), 0, len(X))

    # linear regression on datapoints
    classifier = linear_model.LinearRegression()
    val_x = np.array(range(0, len(X))).reshape(-1, 1)
    val_y = X.reshape(-1, 1)
    classifier.fit(val_x, val_y)

    output = classifier.predict(val_x)

    plt.plot(output, color="blue", linewidth=4)

    plt.show()


def visualize_classifier(classifier, X, y, title=''):

    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)

    # Create a plot
    plt.figure()

    # Specify the title
    plt.title(title)

    # Choose a color scheme for the plot
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()