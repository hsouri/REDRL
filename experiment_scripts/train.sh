#!/usr/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=5,8,9 python tools/train.py \
    Baseline '(False)' \
    Attack_Classifier_Cat '(False)' \
    Attack_Classifier_Pretrained '(True)' \
    GPU_IDs '(0,1)' \
    Eval_Period "(1)" \
    Train_Batch_Size "(512)"  \
    Test_Batch_Size "(512)" \
    KLD_Loss_Lambda "(0.1)" \
    MSE_Loss_Lambda "(10.0)" \
    L2 "(True)" \
    Label_Smoothing "(True)" \
    Label_Smoothing_epsilon "(0.1)" \
    Adversarial_Loss "(True)" \
    Adversarial_Loss_Lambda "(0.1)" \
    Image_Classification_Loss "(True)" \
    Image_Classification_Loss_Lambda "(0.1)" \
    Attack_Type_Triplet_Loss "(False)" \
    DataLoader_Num_Instaces "(4)" \
    Checkpoint_Period "(5)" \
    Num_Workers "(16)" \
    Input_Size "[64,64]" \
    Base_Learning_Rate "(0.00035)" \
    Warmup_Iters "(10)" \
    Warmup_Factor "(0.01)" \
    Num_Attack_Types "(10)" \
    Log_Period "(500)" \
    sub_sample_ratio "(1.0)"