import os
import shutil
import argparse
import json
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 100_000_000

def create_dataset_json (dataset_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer=None):
    """
    Creates the dataset.json file for nnUNetv2 configuration.
    :param dataset_folder: Path to the folder where dataset.json will be stored.
    :param channel_names: Dictionary mapping channel indices to their names.
    :param labels: Dictionary mapping label names to their respective indices.
    :param num_training: Number of training images.
    :param file_ending: File extension for the images.
    :param overwrite_image_reader_writer: Optional configuration for image reader/writer.
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

    print("Successfully created dataset.json file.")

def rename_labels(directory_path, training_case="axones"):
    """
    Renames label images in a directory to a consistent naming convention.
    :param directory_path: Path to the directory containing label images.
    :param training_case: Prefix for the new file names.
    """
    for count, filename in enumerate(sorted(os.listdir(directory_path))):
        new_filename = f"{training_case}_{count + 1:03}.png"
        source = os.path.join(directory_path, filename)
        destination = os.path.join(directory_path, new_filename)
        os.rename(source, destination)
    print("Labels have been successfully renamed.")

def binarize_and_save_label_images(directory_path):
    """
    Binarizes label images and saves them back to the directory.
    :param directory_path: Path to the directory containing label images.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            source_path = os.path.join(directory_path, filename)
            image = Image.open(source_path).convert('L')
            image_array = np.array(image)

            binarized_array = np.where(image_array == 255, 1, 0).astype(np.uint8)
            binarized_image = Image.fromarray(binarized_array)
            binarized_image.save(source_path, format='PNG')
    print("Binarization of labels completed.")

def verify_binary_labels(directory_path):
    """
    Verifies that label images in a directory are binary (i.e., contain only 0s and 1s).
    :param directory_path: Path to the directory containing label images.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            source_path = os.path.join(directory_path, filename)
            image = Image.open(source_path)
            image_array = np.array(image)

            unique_values = np.unique(image_array)
            if not set(unique_values).issubset({0, 1}):  
                print(f"Binarization error detected in {filename}: unique values {unique_values}")

def rename_images(directory_path, file_ending, training_case="axones"):
    """
    Renames training images in a directory to a consistent naming convention.
    :param directory_path: Path to the directory containing training images.
    :param file_ending: File extension of the images to be renamed.
    :param training_case: Prefix for the new file names.
    """
    image_index = 1
    filenames = sorted(os.listdir(directory_path))
    
    for filename in filenames:
        source_path = os.path.join(directory_path, filename)
        if filename.endswith(file_ending):
            new_filename = f"{training_case}_{image_index:03d}_0000{file_ending}"
            destination_path = os.path.join(directory_path, new_filename)
            os.rename(source_path, destination_path)
            image_index += 1        
    print("Training image renaming complete.")

def convert_images_to_png(source_folder, dest_folder):
    """
    Converts images from a source folder to PNG format and saves them in a destination folder.
    :param source_folder: Path to the source folder containing images to convert.
    :param dest_folder: Path to the destination folder where PNG images will be saved.
    """
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, os.path.splitext(filename)[0] + '.png')
        image = Image.open(source_path)
        image = image.convert('L')
        image.save(dest_path, format='PNG')

def move_one_image_for_testing_and_remove_corresponding_label(images_tr_folder, labels_tr_folder, images_ts_folder):
    """
    Moves one training image to the testing folder and removes its corresponding label image.
    :param images_tr_folder: Path to the training images folder.
    :param labels_tr_folder: Path to the training labels folder.
    :param images_ts_folder: Path to the testing images folder.
    """
    images = sorted(os.listdir(images_tr_folder))
    if not images:
        print("No images found in the training folder.")
        return
    
    test_image = images[0]
    test_image_path = os.path.join(images_tr_folder, test_image)
    test_label = test_image.replace("_0000.png", ".png")
    test_label_path = os.path.join(labels_tr_folder, test_label)

    if not os.path.exists(images_ts_folder):
        os.makedirs(images_ts_folder)
    shutil.move(test_image_path, images_ts_folder)
    print(f"Test image {test_image} moved to {images_ts_folder}.")

    if os.path.exists(test_label_path):
        os.remove(test_label_path)
        print(f"Corresponding label {test_label} removed from {labels_tr_folder}.")
    else:
        print(f"Corresponding label {test_label} not found in {labels_tr_folder}.")


def convert_to_nnunet(input_folder, output_folder, channel_names, labels, num_training, file_ending, dataset_id="030", training_case="axones", overwrite_image_reader_writer=None):
    """
    Converts datasets to the nnUNet format, including image renaming and binarization.
    :param input_folder: Path to the input dataset folder containing images and masks.
    :param output_folder: Path to the output nnUNet dataset folder.
    :param channel_names: Dictionary of channel names.
    :param labels: Dictionary of label identifiers.
    :param num_training: Number of training images.
    :param file_ending: File extension for the images.
    :param dataset_id: Identifier for the dataset.
    :param training_case: Name for the training case.
    :param overwrite_image_reader_writer: Optional reader/writer configuration.
    """
    dataset_folder_name = f"Dataset{dataset_id}_{training_case}"
    dataset_folder = os.path.join(output_folder, dataset_folder_name)
    os.makedirs(dataset_folder, exist_ok=True)
    images_tr_folder = os.path.join(dataset_folder, "imagesTr")
    labels_tr_folder = os.path.join(dataset_folder, "labelsTr")
    images_ts_folder = os.path.join(dataset_folder, "imagesTs")
    os.makedirs(images_tr_folder, exist_ok=True)
    os.makedirs(labels_tr_folder, exist_ok=True)
    os.makedirs(images_ts_folder, exist_ok=True)
    images_folder = os.path.join(input_folder, "Images_entrainement")
    labels_folder = os.path.join(input_folder, "Images_annotees")

    if not os.path.exists(images_folder) or not os.listdir(images_folder):
        print(f"The directory {images_folder} does not exist or is empty.")
        return
    if not os.path.exists(labels_folder) or not os.listdir(labels_folder):
        print(f"The directory {labels_folder} does not exist or is empty.")
        return

    print("Converting images and masks to .png format...")
    convert_images_to_png(images_folder, images_tr_folder)
    convert_images_to_png(labels_folder, labels_tr_folder)
    
    print("Binarizing labels...")
    binarize_and_save_label_images(labels_tr_folder)

    print("Verifying binary labels...")
    verify_binary_labels(labels_tr_folder)

    print("Renaming training images...")
    rename_images(images_tr_folder,'.png', training_case)

    print("Renaming label files...")
    rename_labels(labels_tr_folder, training_case)

    print("Moving a test image to 'imagesTs' and removing the corresponding mask...")
    move_one_image_for_testing_and_remove_corresponding_label(images_tr_folder, labels_tr_folder, images_ts_folder)

    create_dataset_json(dataset_folder, channel_names, labels, num_training, file_ending, overwrite_image_reader_writer)
    print("Conversion completed successfully!")

def main():
    default_dataset_id = "030"
    default_training_case_name = "axones"
    channel_names = {"0": "L"}
    labels = {"background": 0, "axons": 1}
    parser = argparse.ArgumentParser(description="Convert a dataset to the nnUNetv2 format")
    parser.add_argument("input_folder", help="Path to the input data folder with images and masks")
    parser.add_argument("output_folder", help="Path to the output nnUNetv2 folder, should be named 'nnUNet_raw'")
    parser.add_argument("--num_training", type=int, help="Number of training images: default is 19", default=19)
    parser.add_argument("--file_ending", help="Image file extension: default is .png", default=".png")
    parser.add_argument("--overwrite_image_reader_writer", help="ReaderWriter optionnel", default=None)
    parser.add_argument("--dataset_id", help="Dataset identifier, default is '030'", default=default_dataset_id)
    parser.add_argument("--training_case", help="Training case name, default is 'axones'", default=default_training_case_name)
    args = parser.parse_args()
    convert_to_nnunet(args.input_folder, args.output_folder, channel_names, labels, args.num_training, args.file_ending, args.dataset_id, args.training_case, args.overwrite_image_reader_writer)

if __name__ == "__main__":
    main()
