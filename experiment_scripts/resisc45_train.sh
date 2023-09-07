#!/usr/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1,3,4,6 python tools/train.py \
    Baseline '(False)' \
    Attack_Classifier_Cat '(False)' \
    Attack_Classifier_Pretrained '(True)' \
    GPU_IDs '(0,1)' \
    Log_Period "(400)" \
    Eval_Period "(1)" \
    Checkpoint_Period "(5)" \
    Train_Batch_Size "(128)"  \
    Test_Batch_Size "(512)" \
    KLD_Loss_Lambda "(0.01)" \
    MSE_Loss_Lambda "(10.0)" \
    L2 "(True)" \
    Num_Classes "(45)" \
    Label_Smoothing "(True)" \
    Label_Smoothing_epsilon "(0.1)" \
    Adversarial_Loss "(True)" \
    Adversarial_Loss_Lambda "(0.1)" \
    Image_Classification_Loss "(False)" \
    Image_Classification_Loss_Lambda "(1.0)" \
    Attack_Type_Triplet_Loss "(False)" \
    DataLoader_Num_Instaces "(4)" \
    Num_Workers "(16)" \
    Input_Size "[256,256]" \
    Base_Learning_Rate "(0.00035)" \
    Warmup_Iters "(10)" \
    Warmup_Factor "(0.01)" \
    Num_Attack_Types "(9)" \
    sub_sample_ratio "(1.0)"