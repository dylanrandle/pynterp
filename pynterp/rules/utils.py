import pandas as pd
import numpy as np
import os

def load_compas(path='../data'):
    """
    loads compas dataset (must be located at `path`)
    performs preprocessing required for Apriori algorithm / decision sets
    """
    df = pd.read_csv(os.path.join(path, 'propublica_data_for_fairml.csv'))
    pctiles = np.percentile(df.Number_of_Priors.values, [25, 50, 75, 90, 95])
    for p in pctiles:
        df['Number_of_Priors_GreaterThan_%d' %p] = df.Number_of_Priors.apply(lambda x: 1 if x > p else 0)
    y = df.score_factor
    X = df.drop(columns=['score_factor', 'Number_of_Priors'], inplace=False)
    return X, y
