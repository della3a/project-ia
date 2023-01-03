from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np


def plot_confusion_matrix(y_test, y_pred, model_name):
   # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for " + model_name)
    plt.colorbar()
    tick_marks = np.arange(len(["depressed", "not depressed"]))
    plt.xticks(tick_marks, ["depressed", "not depressed"], rotation=45)
    plt.yticks(tick_marks, ["depressed", "not depressed"])

    # Add labels to each cell
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def evaluate_model(y_test, y_pred):
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # f1 score
    f1 = f1_score(y_test, y_pred)
    print(f'F1 score: {f1:.2f}')

    # precision score
    precision = precision_score(y_test, y_pred)
    print(f'Precision score: {precision:.2f}')

    # recall score
    recall = recall_score(y_test, y_pred)
    print(f'Recall score: {recall:.2f}')

    return accuracy, f1, precision, recall
