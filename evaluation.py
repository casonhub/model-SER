#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/3/2023
# @File: evaluation.py

import os
import csv

import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from torchmetrics.functional.classification import specificity, accuracy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame


from lightning_model import LigthningAST
from dataset import SpeechRecordingsSpectrogramDataset
from utils import get_metadata, get_dataset_norm_stats, plot_and_save_confusion_matrix

def evaluate(model_paths, dataset_name, n_folds, sample_rate=16000, n_mels=64, n_frames=512, seed=0):
    """
        Evaluate trained model for given dataset. 

        Parameters
        ----------
        model_paths : list(str)
            Path to saved models dermined for evaluation.
        dataset_name : str
            Name of dataset to be used for evaluation.
        n_folds : int
            Number of folds to be split the dataset.
        sample_rate : int
            Sample rate to be applied for recordings loading.
        n_mels : int, default 64
            Number of the mel bands in mel filter bank.
        n_frames : int, default 512
            Number of spectrogram time frames
        seed : int, default 0
            Seed for random state control.

        Returns:
        ----------
            None
    """
    print("----------- STARTING EVALUATION OF TRAINED MODELS -----------")
    seed_everything(0)

    # save model paths to csv if something went wrong
    # prepare head
    head = []
    for i in range(n_folds):
        head.append(f"fold {i+1}")

    print(f"model_paths: {model_paths}")
    evaluation_dir_path = os.path.join("evaluation", dataset_name)
    if not os.path.exists(evaluation_dir_path):
        os.mkdir(evaluation_dir_path)
    with open(os.path.join(evaluation_dir_path, "model_paths.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["model paths"])
        for model_path in model_paths:
            writer.writerow([model_path])

    # get paths to trained model after k-fold cv
    filepaths, labels, encoded_labels = get_metadata(dataset_name)
    labels_mapping = list(set(list(zip(encoded_labels, labels))))
    labels_mapping.sort(key=lambda pair: pair[0])
    sorted_labels = list(map(lambda pair: pair[1], labels_mapping))
    n_classes = np.unique(encoded_labels).shape[0]
    targets_evidence = torch.tensor([])
    preds_evidence = torch.tensor([])

    # split dataset into number of folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    mean, std = get_dataset_norm_stats(dataset_name)

    for fold_idx, (_, test_idx) in enumerate(skf.split(filepaths, encoded_labels)):
        pin_memory=False
        num_workers = 1
        if torch.cuda.is_available():
            pin_memory = True
            num_workers = 4

        test_dataset = SpeechRecordingsSpectrogramDataset(
            filepaths[test_idx],
            encoded_labels[test_idx].astype(np.int64),
            target_signal_sr=sample_rate,
            n_mels=n_mels,
            n_frames=n_frames,
            normalize=True,
            mean=mean,
            std=std
        )

        pin_memory=False
        num_workers = 1
        accelerator = "cpu"
        devices = "auto"

        if torch.cuda.is_available():
            pin_memory = True
            num_workers = 4
            accelerator = "gpu"
            devices = -1

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, pin_memory=pin_memory, num_workers=num_workers)

        model = LigthningAST.load_from_checkpoint(model_paths[fold_idx])
        # put model into evaluation mode
        model.eval()
        trainer = Trainer(accelerator=accelerator, devices=devices, inference_mode=True)
        trainer.test(model, test_dataloader)
        preds_evidence = model.test_preds
        targets_evidence = model.test_targets
        print(f"preds_evidence.shape={preds_evidence.shape}, targets_evidence.shape={targets_evidence.shape}")

    # save report based on the whole predictions
    # check if evaluation folder for given dataset
    evaluation_folder = os.path.join("evaluation", dataset_name)
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)
    report_dict = classification_report(targets_evidence.tolist(), preds_evidence.tolist(), output_dict=True, target_names=sorted_labels)


    # compute specificity
    print(f"preds_evidence: {preds_evidence}")
    print(f"targets_evidence: {targets_evidence}")
    spec = specificity(preds_evidence, targets_evidence, task='multiclass', num_classes=n_classes, average="none")
    spec_macro = specificity(preds_evidence, targets_evidence, task='multiclass', num_classes=n_classes, average="macro")
    spec_weighted = specificity(preds_evidence, targets_evidence, task='multiclass', num_classes=n_classes, average="weighted")
    acc_weighted = accuracy(preds_evidence, targets_evidence, task='multiclass', num_classes=n_classes, average="weighted")

    print(f"SPECIFICITY: {spec}")

    # add specificity to classification report
    for i, class_spec in enumerate(spec):
        report_dict[sorted_labels[i]].update({'specificity': class_spec.item()})
        report_dict['macro avg'].update({'specificity': spec_macro.item()})
        report_dict['weighted avg'].update({'specificity': spec_weighted.item()})
    report_dict.update({'weighted accuracy': acc_weighted.item()})

    DataFrame(report_dict).to_csv(os.path.join(evaluation_folder, "classification_report.csv"))

    # compute confusion matrix and save confusion matrix as a picture
    cm = confusion_matrix(targets_evidence.tolist(), preds_evidence.tolist())
    print(cm)
    plot_and_save_confusion_matrix(cm, n_classes, sorted_labels, os.path.join(evaluation_folder, "confusion_matrix.pdf"))