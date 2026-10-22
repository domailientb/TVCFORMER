#!/usr/bin/env bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=TVCFormer
model_id_name=TVCFormer_weather
yaml_path=configs/weather.yaml
data_name=custom

for pred_len in 96 192 336 720
do
    python -u run_TVCFormer.py \
        --config_yaml $yaml_path \
        --is_training 1 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --pred_len $pred_len >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done
