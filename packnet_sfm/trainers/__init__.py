"""
Trainers
========

Trainer classes providing an easy way to train and evaluate SfM models
when wrapped in a ModelWrapper.

Inspired by pytorch-lightning.

"""

from packnet_sfm.trainers.new_trainer import NewTrainer

__all__ = ["NewTrainer"]
