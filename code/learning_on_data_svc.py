# Import the necessary libraries
from sklearn.svm import SVC
from tools import evaluate_model, plot_confusion_matrix


def learning_on_data_svc(tweets_train, tweets_test, y_train, y_test):

    # Create a SVM classifier
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    # Train the classifier
    clf.fit(tweets_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(tweets_test)

    # Evaluate the model's performance
    accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred, "SVM")

    return clf
