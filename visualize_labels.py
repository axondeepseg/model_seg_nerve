import cv2
import os

def main(images_folder, masks_folder, output_folder):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir les fichiers dans le dossier des images
    for filename in os.listdir(images_folder):
        if filename.endswith(".tif"):  # Assurez-vous que seuls les fichiers d'image TIFF sont traités
            # Charger l'image
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)

            # Vérifier si l'image a été chargée correctement
            if image is None:
                print(f"Erreur : impossible de charger l'image {filename}.")
                continue

            # Construire le chemin vers le masque binaire correspondant
            # Supposer que le nom du masque correspond au nom de l'image avec un suffixe différent
            mask_filename = filename.replace(".tif", "_annotee.png")
            mask_path = os.path.join(masks_folder, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Vérifier si le masque a été chargé correctement
            if mask is None:
                print(f"Erreur : impossible de charger le masque pour {filename}.")
                continue

            # Trouver les contours dans le masque binaire
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dessiner les contours sur l'image originale en noir et avec une épaisseur plus grande
            image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 0, 0), 10)
            jpeg_quality = 10
            # Enregistrer l'image avec les contours sur le disque
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_contours.jpeg")
            cv2.imwrite(output_image_path, image_with_contours, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

            print(f"L'image avec contours {output_image_path} a été enregistrée avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images and masks.')
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--masks_folder', type=str, required=True, help='Path to the folder containing masks')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')

    args = parser.parse_args()

    main(args.images_folder, args.masks_folder, args.output_folder)