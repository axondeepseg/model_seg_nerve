import os
import shutil
import argparse
import json
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 100_000_000

def create_dataset_json (dataset_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer=None):
    """
    Crée le fichier dataset.json pour nnUNetv2.
    """
    dataset_json_path = os.path.join(dataset_folder, "dataset.json")
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

def rename_labels(directory_path, training_case="axones"):
    for count, filename in enumerate(sorted(os.listdir(directory_path))):
        new_filename = f"{training_case}_{count + 1:03}.png"
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

def rename_images(directory_path, file_ending, training_case="axones"):
    # Crée un compteur pour suivre le numéro de l'image
    image_index = 1
    
    # Récupère et trie la liste des fichiers dans le répertoire pour assurer un ordre cohérent
    filenames = sorted(os.listdir(directory_path))
    
    for filename in filenames:
        # Construit le chemin complet vers le fichier source
        source_path = os.path.join(directory_path, filename)
        
        # Vérifie si le fichier est une image et correspond à l'extension désirée
        if filename.endswith(file_ending):
            # Construit le nouveau nom de fichier en suivant le format désiré avec le suffixe "_0000"
            new_filename = f"{training_case}_{image_index:03d}_0000{file_ending}"
            # Construit le chemin complet vers le nouveau fichier
            destination_path = os.path.join(directory_path, new_filename)
            
            # Renomme le fichier
            os.rename(source_path, destination_path)
            
            # Incrémente le compteur pour le prochain fichier
            image_index += 1
            
    print("Renommage des images d'entraînement terminé.")

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

def move_one_image_for_testing_and_remove_corresponding_label(images_tr_folder, labels_tr_folder, images_ts_folder):
    """
    Déplace une image du dossier imagesTr vers imagesTs pour le test, 
    et supprime son masque correspondant dans labelsTr.
    """
    # Obtenir la liste des images d'entraînement
    images = sorted(os.listdir(images_tr_folder))
    if not images:
        print("Aucune image trouvée pour le test.")
        return
    
    # Sélectionner la première image pour le test
    test_image = images[0]
    test_image_path = os.path.join(images_tr_folder, test_image)
    
    # Construire le nom du masque correspondant
    test_label = test_image.replace("_0000.png", ".png")  # Ajustez selon le format de nommage des masques
    test_label_path = os.path.join(labels_tr_folder, test_label)
    
    # Vérifier si le dossier imagesTs existe, sinon le créer
    if not os.path.exists(images_ts_folder):
        os.makedirs(images_ts_folder)

    # Déplacer l'image vers le dossier de test
    shutil.move(test_image_path, images_ts_folder)
    print(f"Image de test {test_image} déplacée vers {images_ts_folder}.")

    # Supprimer le masque correspondant
    if os.path.exists(test_label_path):
        os.remove(test_label_path)
        print(f"Masque correspondant {test_label} supprimé de {labels_tr_folder}.")
    else:
        print(f"Le masque correspondant {test_label} est introuvable dans {labels_tr_folder}.")


def convert_to_nnunet(input_folder, output_folder, channel_names, labels, num_training, file_ending, dataset_id="030", training_case="axones", overwrite_image_reader_writer=None):
    # Assurez-vous que les dossiers de sortie existent
    dataset_folder_name = f"Dataset{dataset_id}_{training_case}"
    dataset_folder = os.path.join(output_folder, dataset_folder_name)
    os.makedirs(dataset_folder, exist_ok=True)
    images_tr_folder = os.path.join(dataset_folder, "imagesTr")
    labels_tr_folder = os.path.join(dataset_folder, "labelsTr")
    images_ts_folder = os.path.join(dataset_folder, "imagesTs")
    os.makedirs(images_tr_folder, exist_ok=True)
    os.makedirs(labels_tr_folder, exist_ok=True)
    os.makedirs(images_ts_folder, exist_ok=True)

    # Dossiers source pour les images et les masques sources
    images_folder = os.path.join(input_folder, "Images_entrainement")
    labels_folder = os.path.join(input_folder, "Images_annotees")

    # Vérifiez l'existence des dossiers source et leur non-vide
    if not os.path.exists(images_folder) or not os.listdir(images_folder):
        print(f"Le dossier {images_folder} n'existe pas ou est vide.")
        return
    if not os.path.exists(labels_folder) or not os.listdir(labels_folder):
        print(f"Le dossier {labels_folder} n'existe pas ou est vide.")
        return

    # Convertir les images et les masques au format .png
    print("Conversion des images et des masques au format .png...")
    convert_images_to_png(images_folder, images_tr_folder)
    convert_images_to_png(labels_folder, labels_tr_folder)
    
    #Binariser les labels
    print("Binarisation des labels...")
    binarize_and_save_label_images(labels_tr_folder)

    # Vérifier que les labels sont binaires
    print("Vérification des labels binaires...")
    verify_binary_labels(labels_tr_folder)

    # Renommer les images d'entraînement dans le dossier 'imagesTr'
    print("Renommage des images d'entraînement...")
    rename_images(images_tr_folder,'.png', training_case)

    # Appeler la fonction de renommage pour les labels dans output_labels_folder
    print("Renommage des fichiers de labels...")
    rename_labels(labels_tr_folder, training_case)

    # Appeler la fonction pour déplacer une image pour le test et supprimer le masque correspondant
    print("Déplacement d'une image vers imagesTs pour le test et suppression du masque correspondant...")
    move_one_image_for_testing_and_remove_corresponding_label(images_tr_folder, labels_tr_folder, images_ts_folder)

    # Créer le fichier dataset.json après la conversion
    create_dataset_json(dataset_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer)
    print("Conversion terminée avec succès !")

def main():
    default_dataset_id = "030"
    default_training_case_name = "axones"
    channel_names = {"0": "L"}
    labels = {"background": 0, "axons": 1}
    parser = argparse.ArgumentParser(description="Conversion d'un ensemble de données en format nnUNetv2")
    parser.add_argument("input_folder", help="Chemin vers le dossier de données d'entrée où se trouvent les images et les masques")
    parser.add_argument("output_folder", help="Chemin vers le dossier de sortie nnUNetv2 qui doit être nommé 'nnUNet_raw")
    parser.add_argument("--num_training", type=int, help="Nombre d'images d'entraînement : par défaut c'est 19", default=19)
    parser.add_argument("--file_ending", help="Extension des fichiers d'image : par défaut c'est .png", default=".png")
    parser.add_argument("--overwrite_image_reader_writer", help="ReaderWriter optionnel", default=None)
    parser.add_argument("--dataset_id", help="Identifiant du dataset, par défaut '030'", default=default_dataset_id)
    parser.add_argument("--training_case", help="Nom du cas d'entraînement, par défaut 'axones'", default=default_training_case_name)
    args = parser.parse_args()
    convert_to_nnunet(args.input_folder, args.output_folder, channel_names, labels, args.num_training, args.file_ending, args.dataset_id, args.training_case, args.overwrite_image_reader_writer)

if __name__ == "__main__":
    main()
