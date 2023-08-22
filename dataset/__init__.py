import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch

from .collate_fn import snapshot_collate_fn
from .data_manager import DataManagerKL
from .enem_dataset import EnemDataset
from .utils import random_split, user_split


def get_data(cfg):
    root_path = cfg.data.root
    path_dict = cfg.data.path_dict
    seed = cfg.random_seed

    # fix seed
    np.random.seed(seed)
    random.seed(seed)

    # users
    if path_dict["unbiased_users"] != None:  # unbiased users
        unbiased_users = np.load(
            os.path.join(root_path, path_dict["unbiased_users"]), allow_pickle=True
        )
    else:
        unbiased_users = np.array([])
    if path_dict["biased_users"] != None:  # biased users
        biased_users = np.load(
            os.path.join(root_path, path_dict["biased_users"]), allow_pickle=True
        )
    else:
        biased_users = np.array([])
    test_users = np.load(
        os.path.join(root_path, path_dict["test_users"]), allow_pickle=True
    )

    # interactions
    org_df = pd.read_csv(os.path.join(root_path, path_dict["original_interactions"]))

    df = pd.read_csv(os.path.join(root_path, path_dict["interactions"]))

    unbiased_df = df[df["user_id"].isin(unbiased_users)]
    unbiased_train_df, unbiased_val_df = user_split(unbiased_df, cfg.data.split_ratio)
    unbiased_val_train_df, unbiased_val_eval_df = random_split(
        unbiased_val_df, cfg.data.split_ratio
    )

    biased_df = df[df["user_id"].isin(biased_users)]
    biased_train_df, biased_eval_df = random_split(biased_df, cfg.data.split_ratio)

    unbiased_train_biased = pd.concat([biased_df, unbiased_train_df])

    test_df = df[df["user_id"].isin(test_users)]

    # item split for CAT evaluation
    total_items = org_df["item_id"].unique()
    test_items = np.random.choice(
        total_items,
        round(len(total_items) * (1 - cfg.data.test_pool_ratio)),
        replace=False,
    )
    pool_items = list(set(list(total_items)) - set(list(test_items)))
    init_items = []

    biased_pool_df = biased_df[biased_df["item_id"].isin(total_items)]
    biased_feature_df = biased_df[biased_df["item_id"].isin(init_items)]

    test_pool_df = test_df[test_df["item_id"].isin(pool_items)]
    test_feature_df = test_df[test_df["item_id"].isin(init_items)]
    test_eval_df = test_df[test_df["item_id"].isin(test_items)]

    data_dict = {  # original interactions
        "org": org_df,
        # total interactions
        "total": df,
        # unbiased dataset
        "unbiased": unbiased_df,
        "unbiased_train": unbiased_train_df,
        "unbiased_val": unbiased_val_df,
        "unbiased_val_train": unbiased_val_train_df,
        "unbiased_val_eval": unbiased_val_eval_df,
        # biased dataset
        "biased": biased_df,
        "biased_train": biased_train_df,
        "biased_eval": biased_eval_df,
        # mixed dataset
        "unbiased_train+biased": unbiased_train_biased,
        # CAT dataset
        "biased_pool": biased_pool_df,
        "biased_feature": biased_feature_df,
        "test_pool": test_pool_df,
        "test_feature": test_feature_df,
        "test_eval": test_eval_df,
        # users
        "unbiased_users": unbiased_users,
        "biased_users": biased_users,
        "test_users": test_users,
        # items
        "items": total_items,
    }

    # datasets
    datasets = {}
    for key in [
        "unbiased_train",
        "unbiased_val_train",
        "unbiased_val_eval",
        "unbiased_train+biased",
        "biased_train",
        "biased_eval",
    ]:
        datasets[key] = EnemDataset(data_dict[key])

    # dataloaders
    dataloaders = {}
    for key in datasets.keys():
        if len(datasets[key]) != 0:
            dataloaders[key] = torch.utils.data.DataLoader(
                datasets[key],
                batch_size=cfg.train.batch_size,
                num_workers=cfg.num_workers,
                collate_fn=partial(
                    snapshot_collate_fn,
                    key_list=["item_id", "user_id", "is_correct"],
                ),
                shuffle=("train" in key),
            )
    return data_dict, datasets, dataloaders


def get_data_manager(cfg, cat_users, feature_df, pool_df, eval_df):
    data_manager = DataManagerKL(cfg, cat_users, feature_df, pool_df, eval_df)
    return data_manager


def update_data(
    cfg,
    selected_df,
    data_dict,
    datasets,
    dataloaders,
):
    selected_df = selected_df.merge(
        data_dict["org"],
        on=["item_id", "user_id", "is_correct", "part_id"],
    )

    unbiased_train_biased = pd.concat(
        [
            data_dict["unbiased_train"],
            selected_df,
        ]
    )

    data_dict["unbiased_train+biased"] = unbiased_train_biased
    datasets["unbiased_train+biased"] = EnemDataset(data_dict["unbiased_train+biased"])
    dataloaders["unbiased_train+biased"] = torch.utils.data.DataLoader(
        datasets["unbiased_train+biased"],
        batch_size=cfg.train.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=partial(
            snapshot_collate_fn,
            key_list=["item_id", "user_id", "is_correct"],
        ),
        shuffle=True,
    )
    return data_dict, datasets, dataloaders
