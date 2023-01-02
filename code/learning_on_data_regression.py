from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def learning_on_data_regression(tweets_train, tweets_test, y_train, y_test):

    # Create a logistic regression model
    clf = LogisticRegression()

    # Train the model on the training data
    clf.fit(tweets_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(tweets_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return accuracy, clf
