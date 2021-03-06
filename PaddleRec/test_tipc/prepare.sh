#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
# The training params
model_name=$(func_parser_value "${lines[1]}")

if [ ${model_name} == "wide_deep" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/save_wide_deep_model https://paddlerec.bj.bcebos.com/wide_deep/wide_deep.tar
    cd test_tipc/save_wide_deep_model && tar -xvf wide_deep.tar && rm -rf wide_deep.tar && cd ../../
    
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    fi
    
elif [ ${model_name} == "deepfm" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/ https://paddlerec.bj.bcebos.com/deepfm/deepfm.tar
    cd test_tipc && tar -xvf deepfm.tar && rm -rf deepfm.tar && cd ..

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/sample_data/* ./test_tipc/data/infer
    fi
elif [ ${model_name} == "ensfm" ]; then
    # prepare pretrained weights and dataset
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/recall/ensfm/data/* ./test_tipc/data
        echo "demo data ready"
    fi
elif [ ${model_name} == "tisas" ]; then
    # prepare pretrained weights and dataset
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/recall/tisas/data/* ./test_tipc/data
        echo "demo data ready"
    fi

fi
