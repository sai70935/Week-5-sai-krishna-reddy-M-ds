import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(r"C:\Users\SAI29\Downloads\cleaned_churn_data.csv")
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('LR')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn'}, axis=1, inplace=True)
    predictions['Churn'].replace({1: 'Yes', 0: 'No'},
                                            inplace=True)
    return predictions['Churn']


if __name__ == "__main__":
    df = load_data(r"C:\Users\SAI29\Downloads\new_churn_data.csv")
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
