from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def learning_on_data(tweets_train, tweets_test, y_train, y_test):
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training set
    clf.fit(tweets_train, y_train)

    # Test the model on the testing set
    y_pred = clf.predict(tweets_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return accuracy, clf
