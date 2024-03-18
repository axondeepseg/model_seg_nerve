#!/bin/bash
# Ce script configure les variables d'environnement pour nnUNet en prenant des chemins spécifiques en entrée

# Vérifier le nombre d'arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 NNUNET_RAW_DIR NNUNET_PREPROCESSED_DIR NNUNET_RESULTS_DIR"
    exit 1
fi

# Assigner les arguments à des variables
NNUNET_RAW_DIR=$(realpath $1)
NNUNET_PREPROCESSED_DIR=$(realpath $2)
NNUNET_RESULTS_DIR=$(realpath $3)

# Configuration des variables d'environnement
export nnUNet_raw="$NNUNET_RAW_DIR"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED_DIR"
export nnUNet_results="$NNUNET_RESULTS_DIR"

echo "Variables d'environnement configurées :"
echo "nnUNet_raw = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results = $nnUNet_results"
