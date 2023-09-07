#!/usr/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py \
    Dataset "CIFAR10" \
    GPU_IDs '(0,1,2,3)' \
    Generator_Type 'NRP_resG' \
    Generator_Pretrained '(False)' \
    Image_Classifier 'resnet50' \
    Log_Period "(50)" \
    Eval_Period "(2)" \
    Checkpoint_Period "(2)" \
    Train_Batch_Size "(128)"  \
    Val_Batch_Size "(512)" \
    L2 "(False)" \
    Num_Classes "(10)" \
    Num_Attack_Types "(5)" \
    Attack_Classifier 'resnet18' \
    Baseline "(True)" \
    Baseline_Source "IMG" \
    Reconstruction_Loss "(False)" \
    Reconstruction_Loss_Lambda "(0.01)" \
    Perceptual_Loss "(False)" \
    Perceptual_Loss_Lambda "(0.1)" \
    Adversarial_Loss "(False)" \
    Adversarial_Loss_Lambda "(0.01)" \
    Image_Classification_Loss "(False)" \
    Image_Classification_Loss_Lambda "(1.0)" \
    Attack_Classifier_Cat "(True)" \
    Attack_Det_Loss "(True)" \
    Attack_Det_Loss "(1.0)" \
    Num_Workers "(32)" \
    Input_Size "[64,64]" \
    Base_Learning_Rate "(0.00035)" \
    Warmup_Iters "(10)" \
    Steps "(40, 70, 90)" \
    Warmup_Factor "(0.01)" \
    Num_Epochs "(100)"