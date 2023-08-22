import torch

import wandb
from dataset import get_data, get_data_manager, update_data
from IF import UserAIF
from model import get_model
from model.trainer import CAT, train
from utils.config import load_config


def main(cfg):
    torch.manual_seed(cfg.random_seed)

    ## get data
    data_dict, datasets, dataloaders = get_data(cfg)

    if cfg.user_aif:
        print("user aif")
        ## train on unbiased set
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.exp_name + "_unbiased_training",
        )
        model = get_model(
            cfg,
            init_ckpt=None,
        )

        init_ckpt = train(
            cfg,
            model,
            saved_model_name="unbiased_training",
            init_item_param_training=True,
            train_dataloader=dataloaders["unbiased_train"],
            val_train_dataloader=dataloaders["unbiased_val_train"],
            val_eval_dataloader=dataloaders["unbiased_val_eval"],
        )
        wandb.finish()

        ## train biased user's params
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.exp_name + "_biased_training",
        )
        model = get_model(
            cfg,
            init_ckpt=init_ckpt,
        )
        init_ckpt = train(
            cfg,
            model,
            saved_model_name="biased_training",
            init_item_param_training=False,
            train_dataloader=dataloaders["biased_train"],
            val_train_dataloader=None,
            val_eval_dataloader=dataloaders["biased_eval"],
        )
        wandb.finish()

        # de-biasing using UserAIF
        model = get_model(
            cfg,
            init_ckpt=init_ckpt,
        )

        selected_biased_df = UserAIF(
            model, data_dict["biased"], data_dict["unbiased"], data_dict["items"]
        )
        data_dict, datasets, dataloaders = update_data(
            cfg, selected_biased_df, data_dict, datasets, dataloaders
        )

    ## Training
    if cfg.wandb.use_wandb:
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.exp_name + "_final_training",
        )
    model = get_model(
        cfg,
        init_ckpt=None,
    )

    init_ckpt = train(
        cfg,
        model,
        saved_model_name="final_training",
        init_item_param_training=True,
        train_dataloader=dataloaders["unbiased_train+biased"],
        val_train_dataloader=dataloaders["unbiased_val_train"],
        val_eval_dataloader=dataloaders["unbiased_val_eval"],
    )

    ## CAT
    data_manager = get_data_manager(
        cfg,
        cat_users=data_dict["test_users"],
        feature_df=data_dict["test_feature"],
        pool_df=data_dict["test_pool"],
        eval_df=data_dict["test_eval"],
    )

    CAT(cfg, model, data_manager)


if __name__ == "__main__":
    # Model
    model_type = "irt-2pl"

    # Dataset
    dataset_type = "enem"

    # Config
    config_type = "./configs/train.yaml"
    configs = load_config(config_type, model_type, dataset_type)

    # Run
    main(cfg=configs)
