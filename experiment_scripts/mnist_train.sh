#!/usr/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py \
    Dataset "MNIST" \
    GPU_IDs '(0,1)' \
    Generator_Type 'NRP_resG' \
    Image_Classifier 'resnet50' \
    Log_Period "(500)" \
    Eval_Period "(5)" \
    Checkpoint_Period "(5)" \
    Train_Batch_Size "(16)"  \
    Val_Batch_Size "(256)" \
    Perceptual_Loss_Lambda "(10)" \
    Reconstruction_Loss_Lambda "(10.0)" \
    L2 "(True)" \
    Num_Classes "(10)" \
    Adversarial_Loss "(True)" \
    Adversarial_Loss_Lambda "(0.1)" \
    Image_Classification_Loss "(True)" \
    Image_Classification_Loss_Lambda "(1.0)" \
    Num_Workers "(16)" \
    Input_Size "[224,224]" \
    Base_Learning_Rate "(0.00035)" \
    Warmup_Iters "(10)" \
    Warmup_Factor "(0.01)"