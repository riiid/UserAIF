import copy

import torch
from tqdm import tqdm


@torch.no_grad()
def UserAIF(model, biased_df, unbiased_df, items):
    biased_df = calculate_IF(model, biased_df, unbiased_df, items)
    userwise_biased_df = biased_df.groupby("user_id").agg("sum").reset_index()
    userwise_biased_df["IF"] = abs(userwise_biased_df["IF"])
    selected_users = userwise_biased_df[
        userwise_biased_df["IF"] < userwise_biased_df.describe().loc["25%"]["IF"]
    ]["user_id"]
    selected_df = biased_df[biased_df["user_id"].isin(selected_users)]
    print(f"{selected_df.shape[0] / biased_df.shape[0] * 100:.1f}% is selected")
    return copy.deepcopy(selected_df)


@torch.no_grad()
def calculate_IF(model, biased_df, unbiased_df, items):
    ub_users = unbiased_df["user_id"].unique()
    num_ub_users = len(ub_users)
    ub_max_user_id = ub_users.max() + 1

    num_q = len(items)
    item_id2idx = {}
    for i, item_id in enumerate(items):
        item_id2idx[item_id] = i

    print("gradient calculation")
    b_item_id = torch.tensor(biased_df["item_id"].values)
    b_user_id = torch.tensor(biased_df["user_id"].values)
    b_is_correct = torch.tensor(biased_df["is_correct"].values)
    b_data = {
        "item_id": b_item_id,
        "user_id": b_user_id,
        "is_correct": b_is_correct,
    }
    b_prob, theta_i = model(b_data)
    theta_i = theta_i.squeeze(1)

    _grad_L_biased = (b_prob - b_is_correct).unsqueeze(-1) * torch.cat(
        [theta_i, -torch.ones(theta_i.shape[0], 1)], dim=-1
    )
    grad_L_biased = torch.zeros((_grad_L_biased.shape[0], num_q * 2 + num_ub_users))
    for idx, (grad, item_id) in enumerate(zip(_grad_L_biased, b_item_id)):
        grad_L_biased[idx][item_id2idx[item_id.item()]] = grad[0]
        grad_L_biased[idx][num_q + item_id2idx[item_id.item()]] = grad[1]

    print("hessian calculation")
    H_aa = torch.zeros(num_q)
    H_dd = torch.zeros(num_q)
    H_ad = torch.zeros(num_q)
    H_tt = torch.zeros(ub_max_user_id)
    H_at = torch.zeros((ub_max_user_id, num_q))
    H_dt = torch.zeros((ub_max_user_id, num_q))

    for j in tqdm(items, total=len(items)):
        partial_df = unbiased_df[unbiased_df["item_id"] == j]
        train_item_id = torch.tensor(partial_df["item_id"].values)
        train_user_id = torch.tensor(partial_df["user_id"].values)
        train_is_correct = torch.tensor(partial_df["is_correct"].values)
        train_data = {
            "item_id": train_item_id,
            "user_id": train_user_id,
            "is_correct": train_is_correct,
        }

        train_probs, theta_ks = model(train_data)
        theta_ks = theta_ks.squeeze(1)
        a_param = model.a_params(train_item_id[0])

        for idx_theta, (prob, theta_k, user_id) in enumerate(
            zip(train_probs, theta_ks, train_user_id)
        ):
            H_aa[item_id2idx[j]] += prob * (1 - prob) * theta_k[0] * theta_k[0]
            H_dd[item_id2idx[j]] += prob * (1 - prob)
            H_ad[item_id2idx[j]] -= prob * (1 - prob) * theta_k[0]
            H_at[user_id, item_id2idx[j]] = (
                prob - train_is_correct[idx_theta]
            ) + theta_k[0] * a_param * prob * (1 - prob)
            H_dt[user_id, item_id2idx[j]] = -a_param * prob * (1 - prob)

    ## H_tt (diagonal components)
    user_list = unbiased_df["user_id"].unique()
    for k in tqdm(user_list, total=len(user_list)):
        train_unbiased_df = unbiased_df[unbiased_df["user_id"] == k]
        train_item_id = torch.tensor(train_unbiased_df["item_id"].values)
        train_user_id = torch.tensor(train_unbiased_df["user_id"].values)
        train_is_correct = torch.tensor(train_unbiased_df["is_correct"].values)
        train_data = {
            "item_id": train_item_id,
            "user_id": train_user_id,
            "is_correct": train_is_correct,
        }
        train_probs, theta_ks = model(train_data)
        theta_ks = theta_ks.squeeze(1)
        a_params = model.a_params(train_item_id)

        H_tt[k] = (a_params**2 * train_probs * (1 - train_probs)).sum()
    hessian = torch.cat(
        [
            torch.cat(
                [torch.diag(H_aa), torch.diag(H_ad), torch.t(H_at[ub_users])], dim=1
            ),
            torch.cat(
                [torch.diag(H_ad), torch.diag(H_dd), torch.t(H_dt[ub_users])], dim=1
            ),
            torch.cat(
                [H_at[ub_users], H_dt[ub_users], torch.diag(H_tt[ub_users])],
                dim=1,
            ),
        ],
        dim=0,
    )
    hessian /= num_q * num_ub_users
    hessian_inv = torch.linalg.inv(hessian)

    # hessian_inv: [num_questions *2 + num_users, num_questions *2 + num_users]
    # grad_L_biased: [num_biased_interactions, num_params]
    IF_total = -torch.matmul(hessian_inv, grad_L_biased.T).T
    biased_df["IF"] = IF_total.sum(1)
    return biased_df
