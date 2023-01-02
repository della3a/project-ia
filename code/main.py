from data_reader import read_data
from preporcessing_data import preprocess_data
from splitting_data import split_data

from learning_on_data import learning_on_data
from learning_on_data_bis import learning_on_data_bis
from learning_on_data_regression import learning_on_data_regression

from prediciting_on_new_data import predict


def main(path):
    tweets_df = read_data(path)

    X, y = preprocess_data(tweets_df)

    tweets_train, tweets_test, y_train, y_test = split_data(X, y)

    accuracy, clf = learning_on_data_bis(
        tweets_train, tweets_test, y_train, y_test)

    # ask the user to enter a new tweet
    new_tweet = input("Enter a new tweet: ")
    predict(new_tweet, clf)


main('projet-techniques-apprentissage-artificiel/data/sentiment_tweets.csv')
