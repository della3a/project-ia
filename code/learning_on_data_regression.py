from sklearn.linear_model import LogisticRegression
from tools import evaluate_model, plot_confusion_matrix


def learning_on_data_regression(tweets_train, tweets_test, y_train, y_test):

    # Create a logistic regression model
    clf = LogisticRegression(max_iter=10000)

    # Train the model on the training data
    clf.fit(tweets_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(tweets_test)

    # Evaluate the model's performance
    accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Logistic Regression")

    return clf
