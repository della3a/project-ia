import os
import pandas as pd

def save_output(X_test, y_test):
    # Create the output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Create a dataframe with the test data
    df = pd.DataFrame({'X': X_test, 'y': y_test})

    # Save the dataframe to a csv file in the output directory
    df.to_csv('output/test_data.csv', index=False)
