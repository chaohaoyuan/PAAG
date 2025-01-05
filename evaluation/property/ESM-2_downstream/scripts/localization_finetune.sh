#!/bin/bash

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -e|--encoder_path)
      encoder_path="$2"
      shift 2
      ;;
    -d|--dataset)
      dataset="$2"
      shift 2
      ;;
    -b|--batch_size)
      batch_size="$2"
      shift 2
      ;;
    -f|--fixed)
      fixed="$2"
      shift 2
      ;;
    -l|--output_path)
      output_path="$2"
      shift 2
      ;;
    -q|--lr)
      lr="$2"
      shift 2
      ;;
    -d|--dropout)
      dropout="$2"
      shift 2
      ;;
    -w|--weight_decay)
      weight_decay="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

echo "--------------------------------------------------------------------------------------------------------"
echo "------------------    version: fix - ${fixed}  bs - ${batch_size} lr - ${lr}   -------------------------"
echo "--------------------------------------------------------------------------------------------------------"

config=""
if [ "$fixed" = "True" ]; then
  config="./config/downstream_task/PretrainESM2/localization_fix.yaml"
elif [ "$fixed" = "False" ]; then
  config="./config/downstream_task/PretrainESM2/localization_tune.yaml"
else
  echo "option - fiexed error!"
  exit 1
fi

num_class=0
if [ "$dataset" = "BinaryLocalization" ]; then
  num_class=2
elif [ "$dataset" = "DeepLoc" ]; then
  num_class=2
elif [ "$dataset" = "SubcellularLocalization" ]; then
  num_class=10
else
  echo "option - dataset error!"
  exit 1
fi

if [ -z "${dropout}" ]; then
  dropout=0
  echo "dropout not passing in, and set to ${dropout}"
else
  echo "dropout is: ${dropout}"
fi

if [ -z "${weight_decay}" ]; then
  weight_decay=0
  echo "weight_decay not passing in, and set to ${weight_decay}"
else
  echo "weight_decay is: ${weight_decay}"
fi

torchrun --nproc_per_node 4 ./script/run_downstream.py \
--config ${config} --checkpoint ${encoder_path} --dataset ${dataset} --batch_size ${batch_size} --num_class ${num_class} --output_path ${output_path} \
--learning_rate ${lr} --dropout ${dropout} --weight_decay ${weight_decay}