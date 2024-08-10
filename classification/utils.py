import pandas as pd
import numpy as np
from tqdm import tqdm


def sliding_window(data, window_size=5):
    """
    # Example usage
    float_list = [1.0, 2.5, 3.7, 4.2, 5.1, 6.3, 7.8, 8.4, 9.0, 10.2]
    result = sliding_window(float_list)
    print(result)
    [[1.0, 2.5, 3.7, 4.2, 5.1],
    [2.5, 3.7, 4.2, 5.1, 6.3],
    [3.7, 4.2, 5.1, 6.3, 7.8],
    [4.2, 5.1, 6.3, 7.8, 8.4],
    [5.1, 6.3, 7.8, 8.4, 9.0],
    [6.3, 7.8, 8.4, 9.0, 10.2]]
    """
    return [data[i:i+window_size] for i in range(len(data) - window_size + 1)]


def get_sliding_dataframe(filename: str = '../time_series_data/SET_DLY_BBL, 5_d2141.csv') -> pd.DataFrame:
    df = pd.read_csv(filename)
    rate_of_change = np.diff(df.close)
    X = sliding_window(df.close, window_size=5)  # Omit first element because it is derivative.
    Y = sliding_window(rate_of_change, window_size=5)
    ## Make dataset for classification

    result_data = []
    for idx, (x, y) in tqdm(enumerate(zip(X, Y)), total=len(X)):
        vec_x: pd.Series = x
        label: float = y[-1]

        row_data = {}
        # Combine vec_x and label into a single row
        for counter, val in enumerate(vec_x.to_list()):
            row_data[counter] = val
        row_data['label'] = label
        result_data.append(row_data)

    # Create the DataFrame from the collected data
    new_df = pd.DataFrame(result_data)
    return new_df


