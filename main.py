#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 31/1/2023

"""
    Entry file of the program. 
    Parses CLI arguments and runs training and evaluation scripts.
"""

import os
import argparse

import pytorch_lightning as pl
import wandb

from training import train
from evaluation import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "Train and evaluate Audio Spectrogram Transformer for Speech emotion recogntion",
        description = "This program wraps training and evaluation of Audio Spectrogram Transformer \
            neural network model. ",
    )
    parser.add_argument("-d", "--dataset", required=True, choices=["RAVDESS", "EMODB", "EMOVO"], help="Dataset for training and evaluation of model.")
    parser.add_argument("-s", "--seed", default=0, help="Seed for random state control")
    parser.add_argument("-b", "--batch-size", default=32, help="Number of samples in one training batch.")
    parser.add_argument("-e", "--epochs", default=10, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning-rate", default=0.001, help="Training learning rate.")
    parser.add_argument("-sr", "--sampling-rate", default=16000, help="Sampling rate for loading speech recordings.")
    parser.add_argument("-fr", "--frames", default=512, help="Mel Spectrogram number of time frames.")
    parser.add_argument("-mel", "--mel-filter-banks", default=64, help="Size of mel filter bank.")
    parser.add_argument("--folds", default=10, help="Number of training folds.")
    parser.add_argument("--wandb-key", required=True, help="API key for login to wandb.ai account.")
    parser.add_argument("--wandb-project", required=True, help="Project in wandb.ai that run(s) will be logged.")

    args = parser.parse_args()

    # set random seed
    seed = int(args.seed)
    pl.seed_everything(seed, workers=True)
    n_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.learning_rate)
    sample_rate = int(args.sampling_rate)
    mel_filter_banks = int(args.mel_filter_banks)
    frames = int(args.frames)
    folds = int(args.folds)
    wandb_key = args.wandb_key
    wandb_project = args.wandb_project

    wandb.login(key=wandb_key)
    os.environ["CODECARBON_LOG_LEVEL"] = "WARNING"

    # run training and evaluation for rest of the datasets
    model_paths = train(
        dataset_name=args.dataset,
        sample_rate=sample_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_folds=folds,
        seed=seed,
        mel_filter_banks=mel_filter_banks,
        frames=frames,
        wandb_project=wandb_project
    )

    print(f"Model PATHS: {model_paths}")
    evaluate(model_paths, args.dataset, n_folds=folds, n_mels=mel_filter_banks, n_frames=frames, seed=seed)
