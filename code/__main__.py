from data_reader import read_data
from preporcessing_data import preprocess_data
from splitting_data import split_data

from learning_on_data_random_forest import learning_on_data_random_forest
from learning_on_data_svc import learning_on_data_svc
from learning_on_data_regression import learning_on_data_regression
from learning_on_data_anns import learning_on_data_anns


from prediciting_on_new_data import predict, classify_tweets


class Tweet:
    def __init__(self, path):
        tweets_df = read_data(path)

        X, y = preprocess_data(tweets_df)

        tweets_train, tweets_test, y_train, y_test = split_data(X, y)

        # Display a menu to allow the user to choose a learning method
        print('Choose a learning method:')
        print('1. Random Forest')
        print('2. SVC')
        print('3. Logistic Regression')
        print('4. Artificial Neural Networks')
        choice = int(
            input('Enter a number and choose a learning method (1-4): '))

        # Map the user's choice to the corresponding learning function
        learning_funcs = {
            1: learning_on_data_random_forest,
            2: learning_on_data_svc,
            3: learning_on_data_regression,
            4: learning_on_data_anns
        }

        print('you have chosen', learning_funcs[choice])

        clf = learning_funcs[choice](
            tweets_train, tweets_test, y_train, y_test)

        # ask the user to enter multiple tweets
        new_tweets = input('Enter multiple tweets separated by a + : ')

        # Split the tweets by space
        tweets = new_tweets.split('+')

        classify_tweets(tweets, clf)


tweet = Tweet(
    '/Users/manelkfc/Desktop/M1/techniques-d-apprentissage-artificiel/projet-techniques-apprentissage-artificiel/data/sentiment_tweets.csv')
