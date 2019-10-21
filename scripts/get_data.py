# makes a dataframe from a csv in current state

import pandas as pd

def fetch(file_path):

    entire_data = pd.read_csv(
        file_path, encoding='ISO-8859-1', delimiter=',', low_memory=False)

    data_df = pd.DataFrame(entire_data)

    return data_df


def throw(df, land):

    df.to_csv(land, index=None, header=True)
