#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python runners/evaluation_tracking.py \
--pretrained_score_model_path results/ckpts/ScoreNet/scorenet.pth \
--pretrained_energy_model_path results/ckpts/EnergyNet/energynet.pth \
--pretrained_scale_model_path results/ckpts/ScaleNet/scalenet.pth \
--data_path Omni6DPose_ROPE_PATH \
--sampler_mode ode \
--percentage_data_for_test 1.0 \
--batch_size 128 \
--seed 0 \
--result_dir tracking \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.25 \
--dino pointwise \
--num_worker 32 \