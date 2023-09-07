#!/usr/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py \
    Baseline '(False)' \
    Attack_Classifier_Cat '(False)' \
    Attack_Classifier_Pretrained '(True)' \
    GPU_IDs '(0,1)' \
    Test_Batch_Size "(512)" \
    Image_Classification_Loss "(True)" \
    Num_Workers "(16)" \
    Input_Size "[64,64]" \
    Num_Classes "(200)" \
    Num_Attack_Types "(10)"  \
    do_save "(False)"