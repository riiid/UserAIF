import glob
import os

import numpy as np
import torch

import wandb


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(
    cfg,
    model,
    saved_model_name,
    init_item_param_training,
    train_dataloader,
    val_eval_dataloader,
    val_train_dataloader=None,
):
    ## train on unbiased set
    result_path = cfg.log_dir
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    early_stopping = EarlyStopping(
        patience=cfg.train.patience,
        delta=1e-4,
        path=os.path.join(result_path, "{}_best_model.ckpt".format(saved_model_name)),
    )
    for epoch in range(cfg.train.max_epochs):
        # train on unbiased train set
        train_losses = []
        model.change_item_param_training_mode(is_train=init_item_param_training)
        for data in train_dataloader:
            labels = data["is_correct"].float()
            probs, _ = model(data)
            optimizer.zero_grad()
            train_loss = model.compute_loss(probs, labels)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
        train_loss = np.mean(train_losses)
        # validation
        # update user param using unbiased valid_train set
        if val_train_dataloader != None:
            model.change_item_param_training_mode(is_train=False)
            for data in val_train_dataloader:
                labels = data["is_correct"].float()
                probs, _ = model(data)
                optimizer.zero_grad()
                val_train_loss = model.compute_loss(probs, labels)
                val_train_loss.backward()
                optimizer.step()
            val_train_auc = model.get_auc(val_train_dataloader.dataset.df)
        else:
            val_train_loss = 0.0
            val_train_auc = 0.0

        # evaluate the updated user param using unbiased valid_eval set
        with torch.no_grad():
            val_eval_losses = []
            for data in val_eval_dataloader:
                labels = data["is_correct"].float()
                probs, _ = model(data)
                val_eval_loss = model.compute_loss(probs, labels)
                val_eval_losses.append(val_eval_loss.item())
            val_eval_loss = np.mean(val_eval_losses)
            early_stopping(val_eval_loss, model)

        val_eval_auc = model.get_auc(val_eval_dataloader.dataset.df)
        print(
            f"epoch: {epoch} || train_loss: {train_loss:.4f} || val_loss: {val_eval_loss:.4f} || val_auc: {val_eval_auc:.4f}"
        )
        if cfg.wandb.use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_train/loss": val_train_loss,
                    "val_eval/loss": val_eval_loss,
                    "val_train/auc": val_train_auc,
                    "val_eval/auc": val_eval_auc,
                    "epoch": epoch,
                }
            )
        if early_stopping.early_stop:
            print("Early stopping")
            break
    init_ckpt = glob.glob(
        os.path.join(result_path, "{}*.ckpt".format(saved_model_name))
    )[0]
    return init_ckpt


def CAT(cfg, model, data_manager):
    model.init_s_param()
    for acquisition_step in range(1, cfg.cat.num_acquisition + 1):
        print(acquisition_step)

        # acquisition
        data_manager.acquisition(acquisition_step, model)

        # update
        sub_new_data = data_manager.data["feature"]

        model.user_update(sub_new_data)

        # evaluation
        if data_manager.data["eval"].shape[0] != 0:
            eval_auc = model.get_auc(data_manager.data["eval"])
            print(f"[Eval] auc: {eval_auc:.4f}")

            if cfg.wandb.use_wandb:
                wandb.log(
                    {
                        "CAT_step": acquisition_step,
                        "CAT/Auc_eval": eval_auc,
                    }
                )
