#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --account=def-jcohen
#SBATCH --time=3-00:00:00  
#SBATCH --cpus-per-task=10  
#SBATCH --mem=100G           
#SBATCH --gpus-per-node=v100:1  

DATASET_ID="030"

for FOLD in 0 1 2 3 4 
do
    echo "Entrainement du modele pour le FOLD $FOLD"
    nnUNetv2_train $DATASET_ID 2d $FOLD --npz
done
echo "Entrainement termine"
