import numpy as np
import torch
import torch.nn as nn
from scipy import optimize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .loss import bce_loss, bce_loss_multi


class IRTEmbedding(nn.Module):
    def __init__(self, size, constraint=None):
        super().__init__()
        self.size = size
        self.params = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(size)),
            requires_grad=True,
        )
        self.constraint = constraint

    def apply_constraint(self, params, constraint):
        if constraint == "non_negativity":
            params = nn.functional.softplus(params)
        return params

    def get_params(self):
        return self.apply_constraint(self.params, self.constraint)

    def forward(self, x):
        try:
            params = self.apply_constraint(self.params[x], self.constraint)
        except:
            params = self.apply_constraint(
                self.params[x.type(torch.LongTensor)], self.constraint
            )
        return params


class IRT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.base_path = cfg.log_dir
        self.detach_q = False

        self.s_params = IRTEmbedding(
            (self.cfg.common.num_s, self.cfg.common.num_kc, self.cfg.common.num_d),
        )
        self.a_params = IRTEmbedding(
            (self.cfg.common.num_q, self.cfg.common.num_kc, self.cfg.common.num_d),
            constraint="non_negativity",
        )
        self.d_params = IRTEmbedding(
            (self.cfg.common.num_q, self.cfg.common.num_kc, 1),
            # constraint="non_negativity",
        )

        self.bce_loss = nn.BCELoss()

    def change_item_param_training_mode(self, is_train):
        self.detach_q = not is_train

    def forward(self, x):
        u_ids = x["user_id"]
        q_ids = x["item_id"]

        # N * K * D
        skills = self.s_params(u_ids)

        if self.detach_q:
            discrim = self.a_params(q_ids).detach()
            diffi = self.d_params(q_ids).detach()
        else:
            discrim = self.a_params(q_ids)
            diffi = self.d_params(q_ids)

        # N * K * D => N * K
        tmp = discrim * skills
        tmp = torch.sum(tmp, axis=-1)
        logits = tmp - diffi.squeeze(-1)

        # N * K => N
        probs = torch.sigmoid(logits)
        # probs = logits
        prob = torch.prod(probs, dim=-1)

        return prob, skills

    def compute_loss(self, probs, labels):
        loss = self.bce_loss(probs, labels)
        return loss

    def init_s_param(self):
        self.s_params = IRTEmbedding(
            (self.cfg.common.num_s, self.cfg.common.num_kc, self.cfg.common.num_d),
        )

    @torch.no_grad()
    def user_update(self, data):
        s_param = self.s_params
        a_param = self.a_params
        d_param = self.d_params

        users = data["user_id"].values
        items = data["item_id"].values

        discrim = a_param(torch.tensor(items)).squeeze(-1).squeeze(-1)
        diffi = d_param(torch.tensor(items)).squeeze(-1).squeeze(-1)
        sparam = s_param(torch.tensor(users)).squeeze(-1).squeeze(-1)
        label = torch.tensor(data["is_correct"].values)

        success_ratio = []
        unique_users = set(users)
        for u in tqdm(unique_users, total=len(unique_users), desc="param update"):
            idx = users == u
            if self.cfg.common.num_d == 1:
                bnds = [(-2, 2)]
                u_discrims = discrim[idx]
                u_diffis = diffi[idx]
                u_labels = label[idx]
                u_sparam = sparam[idx][0].item()  # [1]

                res = optimize.minimize(
                    bce_loss,
                    x0=u_sparam,
                    args=(u_discrims, u_diffis, u_labels),
                    bounds=bnds,
                )
                success_ratio.append(res.success == True)
                s_param.params[u] = torch.tensor(res.x)
            else:
                bnds = [
                    (v.min().item(), v.max().item())
                    for v in s_param.params.squeeze(1).T
                ]

                u_discrims = discrim[idx].squeeze(1).numpy()  # [N, dim]
                u_diffis = diffi[idx].numpy()  # [N]
                u_labels = label[idx].numpy()  # [N]
                u_sparam = sparam[idx].squeeze(1)[0].numpy()  # [dim]

                res = optimize.minimize(
                    bce_loss_multi,
                    x0=u_sparam,
                    args=(u_discrims, u_diffis, u_labels),
                    bounds=bnds,
                )
                success_ratio.append(res.success == True)
                s_param.params[u] = torch.tensor(res.x)
        print("success ratio:", np.mean(success_ratio))

    @torch.no_grad()
    def get_auc(self, data):
        x = {}
        x["user_id"] = torch.tensor(data["user_id"].values)
        x["item_id"] = torch.tensor(data["item_id"].values)

        prob, _ = self.forward(x)
        auc = roc_auc_score(data["is_correct"].values, prob)

        return auc
