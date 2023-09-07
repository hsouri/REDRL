#!/usr/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0,1 python tools/test.py \
    Dataset "CIFAR10" \
    GPU_IDs '(0,1)' \
    Generator_Type 'NRP_resG' \
    Image_Classifier 'resnet50' \
    Test_Batch_Size "(256)" \
    Num_Classes "(10)" \
    Num_Workers "(16)" \
    Input_Size "[224,224]" \
    TEST_ADV_DATA "path/to/adversarial/data" \
    TEST_CKPT_PATH "path/to/checkpoint"
