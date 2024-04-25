# Harmful Brain Activity Classification (HBAC)

This repo contains my solution for the Kaggle `HMS - Harmful Brain Activity Classification` competition.

This first commit contains my solution and the steps to reproduce, but future
versions will be modified as I extend this project a little bit over the next
couple of months to explore some ideas on interpretability, generalizability,
and to reproduce other competitors solutions. 


### Reproducability

To reproduce all of the steps that I took for my final solution, first clone this
library with:

```
$ git clone https://github.com/ryanirl/hbac.git
$ cd hbac
```

Then, download the [Kaggle dataset](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data)
and then unpack it into `./data/` (such that for example `./data/train_eeg/` is a valid directory). 

Once this has been completed, you'll need to preprocess the dataset by running the
following commands.

> [!NOTE]
> This will take ~25 minutes to complete depending on your hardware.

```
$ preprocess.py -i ./data/ --modality eeg --num_workers 4 
$ preprocess.py -i ./data/ --modality eeg_spectrogram --num_workers 4
$ preprocess.py -i ./data/ --modality spectrogram --num_workers 4
```

Finally, once the data is downloaded and preprocessed, you can start training
the models. Training a single fold takes about 4 hours to train from scratch.
Since the dataset had high variance, and performance greatly improved with
ensembling, I chose to use a 10-fold GroupKFold ensemble for my final solution.
Training was originally performed on a single GTX 1070 8GB and takes ~40 hours
to complete (4 hours per fold times 10 folds). To see the results of each model 
read [my writeup](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492189).

> [!NOTE]
> In retrospect, very similar performance can be achieved by using just the raw 1D
> EEG model without the `mid` and `ekg` information. Additionally, maybe including
> the EEG spectrograms was a bit overkill. 

To train the each model, run the following commands:

```
$ sh scripts/train_eeg.sh
$ sh scripts/train_eeg_spec.sh
$ sh scripts/train_spectrogram.sh
$ sh scripts/train_multimodal.sh
```

**Note:** The above scripts will train a single fold, not 10 folds. If you want to train
your will need to perform the following changes:

```
# Remove these two lines
train_stage_1 0 10
train_stage_2 0 10

# Replace them with this, which will train on all folds.
for i in `seq 0 9`
do
    train_stage_1 "$i" 10
    train_stage_2 "$i" 10
done
```

---

Unless stated otherwise in specific file, the license is Apache 2.0.



