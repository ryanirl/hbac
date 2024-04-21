BASE="configs/base_eeg_spectrogram.yaml"
NAME="eeg_spc_cnn_att"

train_stage_1() { # Train on all of the data.
    fold=$1
    n_folds=$2
    python3 train.py --config $BASE \
        --data.fold $fold \
        --data.n_folds $n_folds \
        --trainer.max_epochs 20 \
        --trainer.output_dir "output/${NAME}_stage_1_fold_${fold}"
}

train_stage_2() { # Only train on 'high quality' data where n_votes > 10.
    fold=$1
    n_folds=$2
    python3 train.py --config $BASE \
        --data.fold $fold \
        --data.n_folds $n_folds \
        --data.count_type "upper" \
        --trainer.max_epochs 10 \
        --trainer.output_dir "output/${NAME}_stage_2_fold_${fold}" \
        --from_pretrained "output/${NAME}_stage_1_fold_${fold}/model_final.pt" 
}

train_stage_1 0 10
train_stage_2 0 10