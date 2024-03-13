import os
import shutil
import argparse
import json
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 100_000_000

def create_dataset_json (output_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer=None):
    """
    Crée le fichier dataset.json pour nnUNetv2.
    """
    dataset_json_path = os.path.join(output_folder, "dataset.json")
    dataset_json_content = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_training,
        "file_ending": file_ending,
    }
    if overwrite_image_reader_writer:
        dataset_json_content["overwrite_image_reader_writer"] = overwrite_image_reader_writer

    with open(dataset_json_path, 'w') as json_file:
        json.dump(dataset_json_content, json_file, indent=2)

    print("Fichier dataset.json créé avec succès.")

def rename_labels(directory_path):
    for count, filename in enumerate(sorted(os.listdir(directory_path))):
        new_filename = f"axones_{count + 1:03}.png"
        source = os.path.join(directory_path, filename)
        destination = os.path.join(directory_path, new_filename)
        os.rename(source, destination)
    print("Renommage des labels terminé.")

def binarize_and_save_label_images(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            source_path = os.path.join(directory_path, filename)
            image = Image.open(source_path).convert('L')
            image_array = np.array(image)

            binarized_array = np.where(image_array == 255, 1, 0).astype(np.uint8)
            binarized_image = Image.fromarray(binarized_array)
            binarized_image.save(source_path, format='PNG')

def verify_binary_labels(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            source_path = os.path.join(directory_path, filename)
            image = Image.open(source_path)
            image_array = np.array(image)

            unique_values = np.unique(image_array)
            if not set(unique_values).issubset({0, 1}):  # Vérifiez si toutes les valeurs sont bien 0 ou 255
                print(f"Erreur de binarisation détectée dans {filename}: valeurs uniques {unique_values}")

def rename_images(directory_path, file_ending):
    # Crée un compteur pour suivre le numéro de l'image
    image_index = 1
    
    # Récupère et trie la liste des fichiers dans le répertoire pour assurer un ordre cohérent
    filenames = sorted(os.listdir(directory_path))
    
    for filename in filenames:
        # Construit le chemin complet vers le fichier source
        source_path = os.path.join(directory_path, filename)
        
        # Vérifie si le fichier est une image et correspond à l'extension désirée
        if filename.endswith(file_ending):
            # Construit le nouveau nom de fichier en suivant le format désiré
            new_filename = f"axones_{image_index:03d}{file_ending}"
            # Construit le chemin complet vers le nouveau fichier
            destination_path = os.path.join(directory_path, new_filename)
            
            # Renomme le fichier
            os.rename(source_path, destination_path)
            
            # Incrémente le compteur pour le prochain fichier
            image_index += 1
            
    print("Renommage terminé.")

def convert_images_to_png(source_folder, dest_folder):
    for filename in os.listdir(source_folder):
        # Chemin de l'image source
        source_path = os.path.join(source_folder, filename)
        # Définir le chemin de destination avec l'extension .png
        dest_path = os.path.join(dest_folder, os.path.splitext(filename)[0] + '.png')
        
        # Ouvrir l'image source
        image = Image.open(source_path)
        
        # Convertir l'image en niveaux de gris ('L') avant de sauvegarder en PNG
        # Si vous ne voulez pas convertir en niveaux de gris, enlevez la ligne suivante
        image = image.convert('L')
        
        # Sauvegarder l'image en format PNG
        image.save(dest_path, format='PNG')

def convert_to_nnunet(input_folder, output_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer=None):
    # Assurez-vous que les dossiers de sortie existent
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)
    dossier_labels = os.path.join(output_folder, "labelsTr")
    # Dossiers source pour les images et les masques
    images_folder = os.path.join(input_folder, "Images_entrainement")
    labels_folder = os.path.join(input_folder, "Images_annotees")

    # Vérifiez l'existence des dossiers source et leur non-vide
    if not os.path.exists(images_folder) or not os.listdir(images_folder):
        print(f"Le dossier {images_folder} n'existe pas ou est vide.")
        return
    if not os.path.exists(labels_folder) or not os.listdir(labels_folder):
        print(f"Le dossier {labels_folder} n'existe pas ou est vide.")
        return
    
    # Binariser directement les labels dans 'labels_folder' avant toute autre opération
    print("Binarisation des labels...")
    binarize_and_save_label_images(dossier_labels)

    # Convertir les images et les masques au format .tif
    #print("Conversion des images et des masques au format .png...")
    #convert_images_to_png(images_folder, os.path.join(output_folder, "imagesTr"))
    #convert_images_to_png(labels_folder, os.path.join(output_folder, "labelsTr"))
    
    # Vérifier que les labels sont binaires
    print("Vérification des labels binaires...")
    verify_binary_labels(os.path.join(output_folder, "labelsTr"))

    # Appeler la fonction de séparation des canaux et de renommage pour les images
    #print("Traitement et séparation des canaux des images...")
    #separate_channels_and_rename(os.path.join(input_folder, "Images_entrainement"), os.path.join(output_folder, "imagesTr"))

    # Renommer les images d'entraînement dans le dossier 'imagesTr'
    #print("Renommage des images d'entraînement...")
    #rename_images(os.path.join(output_folder, "imagesTr"), os.path.join(output_folder, "imagesTr"), '.png')

    # Appeler la fonction de renommage pour les labels dans output_labels_folder
    #print("Renommage des fichiers de labels...")
    #rename_labels(os.path.join(output_folder, "labelsTr"))

    # Créer le fichier dataset.json après la conversion
    #create_dataset_json(output_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer)
    #print("Conversion terminée avec succès !")

def main():
    channel_names = {"0": "L"}
    labels = {"background": 0, "axons": 1}
    parser = argparse.ArgumentParser(description="Conversion d'un ensemble de données en format nnUNetv2")
    parser.add_argument("input_folder", help="Chemin vers le dossier de données d'entrée où se trouvent les images et les masques")
    parser.add_argument("output_folder", help="Chemin vers le dossier de sortie nnUNetv2 qui doit être nommé 'nnUNet_raw")
    parser.add_argument("--num_training", type=int, help="Nombre d'images d'entraînement : par défaut c'est 20", default=20)
    parser.add_argument("--file_ending", help="Extension des fichiers d'image : par défaut c'est .png", default=".png")
    parser.add_argument("--overwrite_image_reader_writer", help="ReaderWriter optionnel", default=None)
    args = parser.parse_args()
    convert_to_nnunet(args.input_folder, args.output_folder, channel_names, labels, args.num_training, args.file_ending, args.overwrite_image_reader_writer)

if __name__ == "__main__":
    main()
