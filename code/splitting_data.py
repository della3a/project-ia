from sklearn.model_selection import train_test_split


def split_data(X, y):
    # Splitting the Data in Training and Testing Sets
    # Data training is 70% of the data and the rest (20%) is for testing

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # rename my train and test data into tweets_train and tweets_test
    tweets_train = X_train
    tweets_test = X_test

    return tweets_train, tweets_test, y_train, y_test
