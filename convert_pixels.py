from PIL import Image
import argparse
import os

def transform_intensities(input_image_path, output_image_path):

    img = Image.open(input_image_path)
    img = img.convert('L')
    pixels = list(img.getdata())
    pixels = [255 if pixel == 1 else pixel for pixel in pixels]
    img.putdata(pixels)
    if os.path.isdir(output_image_path):
        filename = os.path.basename(input_image_path).replace('_0000', '_final')
        output_image_path = os.path.join(output_image_path, filename)
    else:
        output_image_path = output_image_path

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    img.save(output_image_path, 'PNG')

def transform_intensities_in_folder(input_folder_path, output_folder_path):

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for filename in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, filename)
        
        output_image_path = os.path.join(output_folder_path, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            transform_intensities(file_path, output_image_path)
def get_args():
    parser = argparse.ArgumentParser(description="Transform image pixel intensities for an entire folder.")
    parser.add_argument("--input_folder_path", type=str, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder_path", type=str, help="Path to the output folder where transformed images will be saved.")
    return parser.parse_args()

def main():
    args = get_args()
    transform_intensities_in_folder(args.input_folder_path, args.output_folder_path)
if __name__ == "__main__":
    main()