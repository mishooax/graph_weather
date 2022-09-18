from typing import Union
import os
import argparse

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from graph_weather.utils.config import YAMLConfig


def build_pl_logger(config: YAMLConfig) -> Union[WandbLogger, TensorBoardLogger, bool]:
    # init logger
    if config["model:wandb:enabled"]:
        # use weights-and-biases
        return WandbLogger(
            project="GNN-WB",
            save_dir=os.path.join(
                config["output:basedir"],
                config["output:logging:log-dir"],
            ),
        )
    elif config["model:tensorboard:enabled"]:
        # use tensorboard
        return TensorBoardLogger(
            os.path.join(
                config["output:basedir"],
                config["output:logging:log-dir"],
            ),
            name="gnn_train_logs",
        )
    return False


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument("--unet", required=False, action="store_true", help="Use a simple UNet model (no GNN)")
    return parser.parse_args()
