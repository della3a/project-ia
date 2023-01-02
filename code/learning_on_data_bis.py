# Import the necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from transformers import BertModel, BertTokenizer
import torch


def learning_on_data_bis(tweets_train, tweets_test, y_train, y_test):

    # Create a SVM classifier
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    # Train the classifier
    clf.fit(tweets_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(tweets_test)
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return accuracy, clf
