from sklearn.ensemble import RandomForestClassifier
from tools import evaluate_model, plot_confusion_matrix


def learning_on_data_random_forest(tweets_train, tweets_test, y_train, y_test):
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training set
    clf.fit(tweets_train, y_train)

    # Test the model on the testing set
    y_pred = clf.predict(tweets_test)

    # Evaluate the model's performance
    accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Random Forest")

    return clf
