# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 31/1/2023
# @File: base_augmentation.py

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torchaudio

from dataset import SpeechRecordingsSpectrogramDataset

def load_model(checkpoint_path):
    try:
        from lightning_model import LigthningAST 
        model = LigthningAST.load_from_checkpoint(checkpoint_path)
        model.eval()
        print(f"Model successfully loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_audio(audio_path, sample_rate=16000, n_mels=64, n_frames=1024):
    print(f"Processing audio file: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    print(f"Original sample rate: {sr}")
    
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        print(f"Resampled to {sample_rate} Hz")
    
    print(f"Waveform shape: {waveform.shape}")
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=int(sample_rate / n_frames)
    )(waveform)
    
    print(f"Mel spectrogram shape before adjustment: {mel_spectrogram.shape}")
    
    if mel_spectrogram.size(-1) < n_frames:
        pad_amount = n_frames - mel_spectrogram.size(-1)
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_amount))
        print(f"Padded mel spectrogram to {n_frames} frames")
    elif mel_spectrogram.size(-1) > n_frames:
        mel_spectrogram = mel_spectrogram[:, :, :n_frames]
        print(f"Truncated mel spectrogram to {n_frames} frames")
    
    print(f"Final mel spectrogram shape: {mel_spectrogram.shape}")
  
    if len(mel_spectrogram.shape) == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
 
    print(f"Final tensor shape: {mel_spectrogram.shape}")
    return mel_spectrogram



def get_metadata(dataset_name):
    """
        Evaluate trained model for given dataset. 

        Parameters
        ----------
        dataset_name : str
            Name of dataset to load its metadata.

        Returns:
        ----------
        filepaths : list(str)
            Filepaths to recordings of selected dataset

        labels : list(str)
            List of recordings labels

        labels : list(int)
            List of encoded labels
    """
    metadata_filepath = os.path.join(os.getcwd(), "datasets", dataset_name, f"{dataset_name}_metadata.csv")
    if dataset_name == "RAVDESS":
        if not os.path.exists(metadata_filepath):
            create_ravdess_metadata_csv()
    elif dataset_name == "EMODB":
        if not os.path.exists(metadata_filepath):
            create_emodb_metadata_csv()
    elif dataset_name == "EMOVO":
        if not os.path.exists(metadata_filepath):
            create_emovo_dataset_metadata_csv()
    else:
        print(f"{dataset_name} DATASET IS NOT IN THE LIST!")
        return None
    with open(metadata_filepath, "r") as f:
        metadata = csv.reader(f)
        # skip head
        next(metadata)
        metadata = np.array(list(metadata))
        return metadata[:, 0], metadata[:, 1], metadata[:, 2].astype(np.int64)


def create_ravdess_metadata_csv():
    """
        Creates metadata csv file with recordings filepaths, labels and encoded labels for RAVDESS dataset

        Parameters
        ----------
        None

        Returns
        ----------
        None
    """
    defaultCodes2labels = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
    labels = list(defaultCodes2labels.values())
    label2id = {}
    for i in range(len(labels)):
        label2id.update({labels[i]: i})
    # fields = ["recording path", "label", "encoded_label"]
    rows = []
    dataset_name = "RAVDESS"
    dataset_path = os.path.join("datasets", "RAVDESS")
    for actor in os.listdir(dataset_path):
        if not actor.startswith("Actor"):
            continue
        for recording in os.listdir(os.path.join(dataset_path, actor)):
            splitted_filename = recording.split("-")
            label = defaultCodes2labels.get(splitted_filename[2])
            rows.append([os.path.join(dataset_path, actor, recording), label, label2id[label]])
    with open(os.path.join(os.getcwd(), "datasets", dataset_name, f"{dataset_name}_metadata.csv"), "w") as f:
        write = csv.writer(f)
        write.writerow(["recording path", "label", "encoded_label"])
        write.writerows(rows)



def create_emodb_metadata_csv():
    """
        Creates metadata csv file with recordings filepaths, labels and encoded labels for Emo-DB dataset

        Parameters
        ----------
        None

        Returns
        ----------
    """   
    # name format - NNTTTEV
    # NN  - speaker number, e.g. 01,02
    # TTT - text code, e.g. a01
    # E   - emotion abbrevation as first letter of emotion label in german, e.g. F - Freude, W - Ärger (Wut)
    # V   - version, e.g. a,b,c
    lettersToLabels = {'W': "anger", 'L': "boredom", 'E': "disgust", 'A': "fear", 'F': "hapiness", 'T': "sadness", 'N': "neutral"}
    labels = list(lettersToLabels.values())
    labels2id = {}
    for i in range(len(labels)):
        labels2id.update({labels[i]: i})
    rows = []
    dataset_path = os.path.join("datasets", "EMODB")
    for recording in os.listdir(dataset_path):
        splitted_filename = recording.split(".")
        if len(splitted_filename) < 2 or splitted_filename[1] != "wav":
            print(f"NOT RECORDING: {recording}")
            continue
        # print(f"recording name: {recording}, recording emotion: {recording[5]}")
        label = lettersToLabels[recording[5]]
        rows.append([os.path.join(dataset_path, recording), label, labels2id[label]])

    with open(os.path.join(dataset_path, "EMODB_metadata.csv"), "w") as f:
        write = csv.writer(f)
        write.writerow(["recording path", "label", "encoded_label"])
        write.writerows(rows)


def create_emovo_dataset_metadata_csv():
    """
        Creates metadata csv file with recordings filepaths, labels and encoded labels for EMOVO dataset

        Parameters
        ----------
        None

        Returns
        ----------
    """ 
    dataset_path = os.path.join("datasets", "EMOVO")
    acronymToLabels = {
        'dis': "disgust",
        'gio': "joy",
        'pau': "fear",
        'rab': "anger",
        'sor': "surprise",
        'tri': "sad",
        'neu': "neutral"
    }
    labels = list(acronymToLabels.values())
    labels2id = {}
    for i in range(len(labels)):
        labels2id.update({labels[i]: i})
    rows = []
    for actor in os.listdir(dataset_path):
        actor_dirpath = os.path.join(dataset_path, actor)
        print(f"actor_dirpath: {actor_dirpath}")
        for recording in os.listdir(actor_dirpath):
            recording_filepath = os.path.join(actor_dirpath, recording)
            splitted_filename = recording.split("-")[0]
            label = acronymToLabels[splitted_filename]
            rows.append([recording_filepath, label, labels2id[label]])
    print(f"dataset_path: {dataset_path}")
    with open(os.path.join(dataset_path, "EMOVO_metadata.csv"), "w") as f:
        write = csv.writer(f)
        write.writerow(["recording path", "label", "encoded_label"])
        write.writerows(rows)

def get_dataset_norm_stats(dataset_name):
    """
        Obtain mean and standard deviation for normalization of dataset

        Parameters
        ----------
        dataset_name : str
            Name of dataset for which values should be obtained

        Returns
        ----------
        mean : float
            mean value of dataset
        std : float
            standard deviation value for dataset
    """
    mean, std = None, None
    with open("datasets_stats.csv") as f:
        reader = csv.reader(f)
        # skip head
        next(reader)
        # get stats based on dataset name
        dataset_stats = list(filter(lambda stats_row: stats_row[0] == dataset_name, list(reader)))[0]
        # print(dataset_stats)
        mean, std = float(dataset_stats[1]), float(dataset_stats[2])

    return mean, std


def compute_dataset_norm_stats(dataset_name, sample_rate, n_frames, n_mels):
    """
        Compute normalization values for demanded dataset

        Parameters
        ----------
        dataset_name : str
            Name of dataset for which values should be computed

        sample_rate : int
            Sampling rate for loading of recordings

        n_frames : int
            Number of mel spectrogram frames

        n_mels : int
            Number of mel spectrogram mel bands

        Returns
        ----------
        None
    """
    filepaths, _, encoded_labels = get_metadata(dataset_name)
    dataset = SpeechRecordingsSpectrogramDataset(filepaths=filepaths, labels=encoded_labels, target_signal_sr=sample_rate, n_frames=n_frames, n_mels=n_mels)
    print(f"dataset len: {dataset.__len__()}")
    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=pin_memory)
    mean, std = [], []
    for inputs, _ in dataloader:
        print(f"curennt inputs shape: {inputs.shape}")
        cur_std, cur_mean = torch.std_mean(inputs)
        mean.append(cur_mean)
        std.append(cur_std)
        # print(cur_mean, cur_std)

    return np.mean(mean), np.mean(std)


def plot_and_save_confusion_matrix(cm, n_classes, class_labels, cm_path):
    """
        Plot and save confusion matrix

        Parameters
        ----------
        cm : list(list(int))
            confusion matrix in form of 2D array

        n_classes : int
            number of considered classes

        class_labels : list(str)
            list of class labels to display in figure

        cm_path : str
            path that determines where to store plot

        Returns
        ----------
        mean : float
            mean value of dataset
        std : float
            standard deviation value for dataset
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # set axis labels and tick marks
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    ax.set_xlabel('Predicted label', fontdict={'fontweight': "semibold"}, labelpad=12.)
    ax.set_ylabel('True label', fontdict={'fontweight': "semibold"}, labelpad=12.)

    # rotate tick labels and set alignment
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor"
    )

    # set threshold for different font color
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_title("Confusion Matrix", fontdict={'fontsize': 16, 'fontweight': "bold"}, pad=18.)
    fig.tight_layout()
    plt.savefig(cm_path, format="pdf")
    plt.show()
