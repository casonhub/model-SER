# **User guide**
This user guide provides short description of program and manual how to execute it. 

## **Description**

This program was implemented for purposes of Master thesis with title *Emotion Recognition from Analysis of a Person’s Speech*. 

It serves for training and evaluation of Audio Spectrogram Transformer neural network for emotion recognition from speech recordings, which was originally proposed by *Gong et al.* [\[paper\]](https://arxiv.org/abs/2104.01778) [\[original repository\]](https://github.com/YuanGongND/ast).

The implementation cosists of two main parts: **training** (module `training.py`) that serves for fitting of the neural network model, and **evaluation** (`evaluation.py`), which perform evaluation of trained model(s) in form of classification metrics computation.

Program is executable through command line interface. It offers list of arguments, which enables managment of model and training hyperparameters.

The entry point of the of the program is `main.py`

---

## **Prerequisities**
  1. Installed `Python` version **3.9**
  2. Installed `Conda` version **22.11**
  3. Create account at https://wandb.ai/, the program execution is directly connected to its API and **it is not possible to run program without it!**.
  4. Download dataset that you want you use (implemetation considers RAVDESS, Emo-DB or EMOVO) and store it to the `dataset/<dataset name>` in **ORIGINAL STRUCTURE** -> very important, functions that create datasets metadata files are dependent on that structure.
    * For example for RAVDESS dataset, the program expects that its file are stored in folder `datasets/RAVDESS`. The innter structure should cotain folders `Actor_01` - `Actor_24` that contains recordings  

---

## **Environment preprartion and program execution**
  1.  ```$ conda env create -n <env name> -f environment.yml``` - create `conda` virtual environment with name `<name>` and all required dependecies (for installation on mac device replace `environment-mac.yml` with `environment-mac.yml`)

  2. ```$ conda activate <env name>``` - activate conda environment with name `<name>`

  3. ```$ python3.9 main.py``` - to start program execution from CLI    
    * arguments list:
      * `-d, --dataset`  - Dataset for training and evaluation of model. *Required, choose one from options RAVDESS, EMODB EMOVO* 
      * `-s, --seed`     -  Seed for random state control. Default value 0. 
      * `-b, --batch-size` -  Number of samples in one training batch. Default value *32*.
      * `-e, --epochs`    -  Number of training epochs. Default to *200*.
      * `-lr, --learning-rate` - Learning rate. Default value *0.0001* 
      * `-sr,  --sampling-rate` - Sampling rate for loading speech recordings. Default value *16000*
      * `-fr,  --frames` - Mel Spectrogram number of time frames. Default value *512*
      * `-mel, --mel-filter-banks` - Size of mel filter bank. Default value *64*
      * `--folds`        - Number of folds for k-fold cross validation. Default value *10*
      * `--wandb-key`    - API key for login to wandb.ai account. *Required*.
      * `--wandb-project` - Project in wandb.ai to log run(s). *Required*.
    
  * example for running 10-fold cross validatiom for RAVDESS dataset, with 200 epochs, lr 0.00001, number of time frames equal to 1024 and mel filter bank with 128 filters:     
  ```$ python3.9 main.py -d RAVDESS -e 200 --folds 10 -lr 0.00001 --mel-filter-banks 64 --frames 1024 --wandb-project=my-awesome-project --wandb-key=b3a6d3e3a65b3d06cbd8b1599183c35758ebac44```


### Trained models and evaluation 
  * trained models are stored in folder `trained_models/<dataset name>`
  * data from evaluation are stored in folder `evaluation/<dataset name>` (paths to saved models, confusion matrix plot, csv file with metrics),



## Other notes
  * implementation was tuned on machines with operation system Linux (with available cuda - installed PyTorch version for cuda) and MacOS Monterey, 
  * for windows machines is not provided the `environment-win.yml`, easy to add, just install packages that are missing :)