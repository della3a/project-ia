from sklearn.neural_network import MLPClassifier
from tools import evaluate_model, plot_confusion_matrix


def learning_on_data_anns(tweets_train, tweets_test, y_train, y_test):
    # Create a neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10000)

    # Train the model on the training set
    clf.fit(tweets_train, y_train)

    # Test the model on the testing set
    y_pred = clf.predict(tweets_test)

    # Evaluate the model's performance
    accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred, "ANNs")

    return clf
