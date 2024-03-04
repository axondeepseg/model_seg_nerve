import os
import shutil
import argparse

def convert_to_nnunet(input_folder, output_folder):
    # Créer le dossier de sortie s'il n'existe pas déjà
    os.makedirs(output_folder, exist_ok=True)

    # Copier les images d'entraînement dans le dossier 'imagesTr'
    images_folder = os.path.join(input_folder, "Images_entrainement")
    output_images_folder = os.path.join(output_folder, "imagesTr")
    shutil.copytree(images_folder, output_images_folder)

    # Copier les masques d'entraînement dans le dossier 'labelsTr'
    labels_folder = os.path.join(input_folder, "Images_annotees")
    output_labels_folder = os.path.join(output_folder, "labelsTr")
    shutil.copytree(labels_folder, output_labels_folder)

    print("Conversion terminée avec succès !")

def main():
    parser = argparse.ArgumentParser(description="Conversion d'un ensemble de données en format nnUNetv2")
    parser.add_argument("input_folder", help="Chemin vers le dossier de données")
    parser.add_argument("output_folder", help="Chemin vers le dossier de sortie nnUNetv2")
    args = parser.parse_args()

    convert_to_nnunet(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
