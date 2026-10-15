"""Hydra CLI dispatching experiment assembly to an explicit training recipe."""

import hydra
from omegaconf import DictConfig

from ..logging import configure_console_logging, log_to_console
from .config import ExperimentConfig, register_experiment_configs, validate_experiment
from .trainer import ILTrainer


def build_trainer(
    config: DictConfig | ExperimentConfig, *, log_callback=None
) -> ILTrainer:
    cfg = validate_experiment(config)
    builder = hydra.utils.get_method(cfg.recipe)
    return builder(cfg, log_callback=log_callback)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train")
def main(config: DictConfig):
    logger = configure_console_logging()
    trainer = build_trainer(config, log_callback=log_to_console)
    logger.info("Training %s on %s", config.name, trainer.device)
    trainer.fit()


if __name__ == "__main__":
    register_experiment_configs()
    main()
