# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: base_augmentation.py

import os
from shutil import copy

import numpy as np
import torch
import pytorch_lightning as pl
import wandb
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from lightning_model import LigthningAST
from utils import get_metadata, get_dataset_norm_stats
from dataset import SpeechRecordingsSpectrogramDataset

from augmentation import ComposeAugmentations, TimeMaskingSpecAugmentation, FreqMaskingSpecAugmentation, RandomSpeedChangeAugmentation, RandomWhiteNoiseAugmentation, RIRAugmentation


def train(dataset_name, sample_rate, n_epochs, batch_size, lr, n_folds, seed, frames, mel_filter_banks, wandb_project):
    """
        Function that performs training of AST model. It prepares the dataset and training loop, which is then exectued

        Parameters
        ----------
        dataset_name : str
            Dataset name to be used in training

        sample_rate : int
            Sampling rate for loading of recordings

        n_epochs : int
            Number of iterations of train loop

        batch_size : int
            Mini-batch size

        lr : float
            Learning rate for neural network parameters update

        n_folds : int
            number of folds, which should be considered for training

        seed : int
            seed for random state control

        frames : int
            Number of time frames of mel spectrogram

        mel_filter_banks : int 
            Size of mel filter bank

        wandb_project : str
            Name of wandb project to store training metrics

        Returns
        ----------
        model_paths : list(str)
            List of paths, where are trained model stored.
    """
    print("-------------- HYPERPARAMETERS ----------------")
    print(f"DATASET NAME={dataset_name}, SAMPLE_RATE={sample_rate}, n_epochs={n_epochs}, batch_size={batch_size} lr={lr}, seed={seed}, frames={frames}, mel_filter_banks={mel_filter_banks}, folds={n_folds}")
    # print("Computing mean and std for normalization purposes...")
    model_size = "tiny224"
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    filepaths, _, encoded_labels = get_metadata(dataset_name)
    n_classes = np.unique(encoded_labels).shape[0]

    model_paths = []

    num_workers = 4
    pin_memory = False
    print(f"cuda availalable={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        pin_memory = True

    # load mean and std from csv file
    mean, std = get_dataset_norm_stats(dataset_name)
    print(f"Normalization values: mean={mean}, std={std}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, encoded_labels)):
        # trainer = None
        print(f"Running fold {fold_idx+1}")
        augumentation_prob = 0.25
        signal_augumentation = ComposeAugmentations([
            RandomWhiteNoiseAugmentation(augumentation_prob, sample_rate),
            RandomSpeedChangeAugmentation(augumentation_prob, sample_rate),
            RIRAugmentation(augumentation_prob, sample_rate)
        ])

        spectrogram_augumentation = ComposeAugmentations([
            TimeMaskingSpecAugmentation(augumentation_prob, sample_rate),
            FreqMaskingSpecAugmentation(augumentation_prob, sample_rate)
        ])

        train_dataset = SpeechRecordingsSpectrogramDataset(
            filepaths[train_idx],
            encoded_labels[train_idx].astype(np.int64),
            sample_rate,
            n_frames=frames,
            n_mels=mel_filter_banks,
            signal_augumentation=signal_augumentation,
            spectrogram_augumentation=spectrogram_augumentation,
            normalize=True,
            mean=mean,
            std=std
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

        wandb.init(project=wandb_project, name=f"{model_size}-{dataset_name}={fold_idx+1}")
        # checkpoints logging dir
        checkpoints_logging_dir = os.path.join(wandb.run.dir, "checkpoints")

        model_checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=checkpoints_logging_dir,
            filename="%s-model-fold-%s-{epoch:02d}-{train_loss:.4f}" % (dataset_name, fold_idx+1),
            save_top_k=1,
            mode='min',
            every_n_epochs=1,
        )

        pl_model = LigthningAST(
            n_frames=frames,
            n_mels=mel_filter_banks,
            n_classes=n_classes,
            model_size=model_size,
            lr=lr,
            n_epochs=n_epochs,
            wandb_logger=True,
        )

        # if use_wandb:
        wandb_logger = WandbLogger(project="AST-SER", log_model="all")

        wandb_logger.experiment.config.update({
            "dataset_name": dataset_name,
            "seed": seed,
            "n_folds": n_folds,
            "batch_size": batch_size
        })
        accelerator = "cpu"
        devices = "auto"
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = -1

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[model_checkpoint_callback],
            accelerator=accelerator,
            devices=devices,
            logger=wandb_logger,
            inference_mode=True,
        )

        # fit the model
        trainer.fit(pl_model, train_loader)

        # if use_wandb:
        wandb.finish()

        # store path of the best model into variable
        src_path = model_checkpoint_callback.best_model_path
        trained_model_name = os.path.split(src_path)[-1]
        dst_path = os.path.join("trained_models", dataset_name, trained_model_name)
        print(f"trained_model_name: {trained_model_name}")
        # copy best model from current fold to "./trained_models/*dataset_name*/*trained_model_name*"
        if not os.path.exists(os.path.join("trained_models", dataset_name)):
            os.makedirs(os.path.join("trained_models", dataset_name))

        copy(src_path, dst_path)
        # add path to model paths evidence for future evaluation
        model_paths.append(dst_path)

    # save emission to csv file
    evaluation_dir_path = os.path.join("evaluation", dataset_name)
    if not os.path.exists(evaluation_dir_path):
        os.makedirs(evaluation_dir_path)

    return model_paths
