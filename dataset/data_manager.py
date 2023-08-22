import os

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import torch
from tqdm import tqdm


def integrand(x, a, d, s):
    p = 1 / (1 + np.exp(-a * (s - d)))
    return (
        p * np.log(p)
        + (1 - p) * np.log(1 - p)
        - (1 - p) * a * d
        + (1 - p) * a * x
        + np.log(1 + np.exp(-a * (x - d)))
    )


class DataManagerKL:
    """
    Class for managing data that changes according to data acquisition.
    """

    def __init__(self, cfg, cat_users, feature_df, pool_df, eval_df=pd.DataFrame([])):
        self.cfg = cfg
        self.cat_users = cat_users
        self.data = {
            "pool": pool_df,
            "feature": feature_df,
            "eval": eval_df,
        }
        self.total_num_acquistion = cfg.cat.num_acquisition
        self.dim = self.cfg.model.common.num_d
        self.column_names = self.data["pool"].columns

    def acquisition(self, acquisition_step, model):
        users = self.cat_users
        feature_df = self.data["feature"].loc[:, self.column_names]
        pool_df = self.data["pool"].reset_index()

        # select items to recommend for each users
        chosen_interactions = []
        for user in tqdm(users, total=len(users), desc="acquisition"):
            user_pool_df = pool_df[pool_df["user_id"] == user].loc[:, self.column_names]

            inputs = {}
            inputs["user_id"] = torch.tensor(
                user_pool_df.loc[:, "user_id"].values
            ).type(torch.LongTensor)
            inputs["item_id"] = torch.tensor(
                user_pool_df.loc[:, "item_id"].values
            ).type(torch.LongTensor)
            unique_items = user_pool_df.loc[:, "item_id"].unique()

            output, _ = model(inputs)

            ### uni-dim
            if self.dim == 1:
                a_param = (
                    model.a_params(torch.tensor(user_pool_df.loc[:, "item_id"].values))
                    .cpu()
                    .detach()
                    .numpy()
                    .squeeze((1, 2))
                )
                d_param = (
                    model.d_params(torch.tensor(user_pool_df.loc[:, "item_id"].values))
                    .cpu()
                    .detach()
                    .numpy()
                    .squeeze((1, 2))
                )
                user_emb = (
                    model.s_params(torch.tensor(user)).cpu().detach().numpy().squeeze()
                )

                delta = 2 / np.sqrt(acquisition_step)
                lower_bd = user_emb - delta
                upper_bd = user_emb + delta
                klinfo = []

                for ind, item in enumerate(unique_items):
                    kl = integrate.quad(
                        integrand,
                        lower_bd,
                        upper_bd,
                        args=(a_param[ind], d_param[ind], user_emb),
                    )
                    klinfo.append(kl[0])
                index = np.argmax(klinfo)
            ### multi-dim
            else:
                output = output.detach().numpy()  # [100, ]

                a_param = (
                    model.a_params(torch.tensor(user_pool_df.loc[:, "item_id"].values))
                    .cpu()
                    .detach()
                    .numpy()
                    .squeeze(1)
                )

                klinfo = (
                    (
                        2 ** (self.dim - 1)
                        / 3
                        * (3 / np.sqrt(acquisition_step)) ** (self.dim + 2)
                    )
                    * (1 - output)
                    * output
                    * np.sum(a_param**2, axis=1)
                )
                index = np.argmax(klinfo)

            chosen_interactions.append(user_pool_df.values[index])

        # update
        chosen_df = pd.DataFrame(
            np.stack(chosen_interactions), columns=feature_df.columns
        )
        self.update(chosen_df)

        # log
        self.log(self.data, acquisition_step)

    def update(self, chosen_df):
        feature_df = self.data["feature"]
        pool_df = self.data["pool"].reset_index()

        updated_dfs = {}
        removed_indices = pd.merge(
            pool_df.reset_index(), chosen_df, on=list(self.column_names)
        )["level_0"]
        updated_dfs["pool"] = (
            pool_df.drop(removed_indices)
            .reset_index(drop=True)
            .loc[:, self.column_names]
        )

        if len(feature_df) != 0:
            feature_df.index += max(feature_df.index)
        updated_dfs["feature"] = pd.concat([feature_df, chosen_df]).loc[
            :, self.column_names
        ]
        updated_dfs["recent_feature"] = chosen_df.loc[:, self.column_names]

        for key in updated_dfs.keys():
            self.data[key] = updated_dfs[key]

    def log(self, dfs, step):
        path = self.cfg.log_dir + "/al" + str(step) + "/data"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        for key in [
            "feature",
            "pool",
            "recent_feature",
        ]:
            dfs[key].to_csv(path + "/{}.csv".format(key))
