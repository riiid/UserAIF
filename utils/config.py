import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf


def load_config(config_type: str, model_type: str, dataset_type: str) -> DictConfig:
    config = OmegaConf.load(config_type)
    config = OmegaConf.merge(config, OmegaConf.from_cli())

    config["data"]["type"] = dataset_type
    config["model"]["type"] = model_type

    # log path
    cur_time = datetime.now().replace(microsecond=0).isoformat()
    log_dir = os.path.join(
        "./logs/{}_{}/{}".format(
            config["model"]["type"], config["data"]["type"], config["wandb"]["exp_name"]
        ),
        cur_time,
    )
    config["log_dir"] = log_dir
    config["cur_time"] = cur_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(log_dir, "run_cfg.yaml"))

    return config
