import numpy as np
from tqdm import tqdm


def random_split(df, split_ratio, user_key="user_id"):
    train_idx, val_idx = [], []
    for _, values in tqdm(df.groupby(user_key)):
        indices = list(values.index)
        np.random.shuffle(indices)
        train_idx += indices[: int(len(indices) * split_ratio)]
        val_idx += indices[int(len(indices) * split_ratio) :]
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]

    return train_df, val_df


def user_split(df, split_ratio, user_key="user_id"):
    user_list = df[user_key].unique()

    train_df = df[df[user_key].isin(user_list[: int(len(user_list) * split_ratio)])]
    val_df = df[df[user_key].isin(user_list[int(len(user_list) * split_ratio) :])]

    return train_df, val_df
