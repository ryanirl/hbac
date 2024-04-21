BASE="configs/base_multimodal.yaml"
NAME="multimodal"

train_stage_1() { # Train on all of the data.
    fold=$1
    n_folds=$2
    python3 train.py --config $BASE \
        --data.fold $fold \
        --data.n_folds $n_folds \
        --trainer.max_epochs 20 \
        --trainer.output_dir "output/${NAME}_stage_1_fold_${fold}" \
        --model multimodal_base_pretrained_backbone_entrypoint \
        --model.eeg_ckpt_path "output/eeg_cnn_rnn_att_stage_2_fold_${fold}/model_best_val-loss_g10.pt" \
        --model.eeg_spec_ckpt_path "output/eeg_spc_cnn_att_stage_2_fold_${fold}/model_best_val-loss_g10.pt" \
        --model.spec_ckpt_path "output/spc_cnn_att_stage_2_fold_${fold}/model_best_val-loss_g10.pt"
}

train_stage_2() { # Fine tune the model on 'high quality' data where n_votes > 10.
    fold=$1
    n_folds=$2
    python3 train.py --config $BASE \
        --data.fold $fold \
        --data.n_folds $n_folds \
        --data.count_type "upper" \
        --data.batch_size 16 \
        --model.freeze_backbone false \
        --optimizer.lr 0.0001 \
        --trainer.max_epochs 10 \
        --trainer.output_dir "output/${NAME}_stage_2_fold_${fold}" \
        --from_pretrained "output/${NAME}_stage_1_fold_${fold}/model_final.pt" 
}

train_stage_1 0 10
train_stage_2 0 10