# model_seg_nerve
Nerve area segmentation for axon density estimation in BF images

## Environment Configuration for nnU-Net
This project utilizes nnU-Net for image segmentation. To ensure that nnU-Net operates correctly, setting up the environment variables properly is crucial. The provided script setup_environment_variables.sh will facilitate this process.

### Prerequisites
Before you begin, make sure you have the following installed:
- nnU-Net
- Bash (for Windows users, Git Bash or WSL can be used)

## Setting Up Conda Environment
To replicate the environment needed for this project, we’ve provided an 'environment.yaml' file. This file lists all the necessary dependencies and their versions. Follow these steps to create an identical environment on your machine:
1. Install Conda: If you don’t have Conda installed, download and install Miniconda or Anaconda from the official website.
2. Open a terminal or command prompt.
3. Before creating the Conda environment, install PyTorch and torchvision by finding the appropriate version for your system on the PyTorch website.
4. Navigate to the directory containing the 'environment.yaml' file.
5. Run the following command to create the Conda environment:
```bash
conda env create -f environment.yaml
```
6. Activate the Conda environment:
```bash
conda activate nnunet_env
```
After following these steps, you will have a Conda environment named 'nnunet_env' with all the necessary dependencies installed.

### Script Usage Instructions
1. Obtaint the script : 
clone the repository or directly download the script from the repository.
2. Execute the script :
Open a terminal and navigate to the folder containing 'setup_environment_variables.sh'.
3. Assign execute permissions (if necessary) :
```bash
chmod +x setup_environment_variables.sh
```
4. Run the script with the appropriate paths :
```bash
./setup_environment.sh /path/to/nnUNet_raw /path/to/nnUNet_preprocessed /path/to/nnUNet_results
```
Replace the paths with those corresponding to your directories. The script will set the environment variables for nnU-Net to operate correctly.

#### Note for Windows Users
When running the script on Windows, please use forward slashes (/) or double backslashes (\\) in the paths. 

### Script Details
The script requires three arguments that are the paths to the following directories:
- NNUNET_RAW_DIR : The directory containing the raw data for nnU-Net.
- NNUNET_PREPROCESSED_DIR : The directory containing the preprocessed data for nnU-Net.
- NNUNET_RESULTS_DIR : The directory containing the results of nnU-Net.
The script first checks that the  three required arguments are provided. It then uses the 'realpath' command to resolve absolute paths, which is important to avoid errors related to relative paths.

The environment variables 'nnUNet_raw', 'nnUNet_preprocessed', and 'nnUNet_results' are then configured to point to the specified directories. These variables are essential for nnU-Net to know where to find and store data at different stages of the segmentation process.

Once the script is executed, it displays the configured paths for confirmation.

## Dataset Preparation for nnU-Net
The Python script 'convert_to_nnunetv2_format.py' is designed to automate the preparation of image data for nnU-Net segmentation tasks. It performs various tasks such as converting images to the appropriate format, renaming them according to nnU-Net conventions, binarizing label images, and creating a dataset.json file required by nnU-Net.

### Using the Script
To use the script, you will need to pass the path to your input data folder containing images and masks, and the path to the output folder where the nnU-Net-ready dataset will be created (nnUNet_raw).
For example:
```bash
python convert_to_nnunetv2_format.py /path/to/input_data /path/to/output_nnunet_data --num_training=19 --file_ending=.tif --dataset_id=030 --training_case=axons
```
Detailed Argument Descriptions
input_folder: Path to your dataset containing the training images and annotations.
output_folder: Where the nnU-Net formatted dataset should be saved (nnUNet_raw).
--num_training: (Optional) The number of training images in your dataset.
--file_ending: (Optional) The file extension of your training images.
--dataset_id: (Optional) A unique identifier for your dataset.
--training_case: (Optional) A name for the training case, which will be used in naming the files.
--overwrite_image_reader_writer: (Optional) If you have a custom image reader/writer configuration for nnU-Net, specify it here.
After running this script, your dataset will be structured in a way that is compatible with nnU-Net, and you will be ready to begin training your segmentation model.

## Training nnU-Net on SLURM Cluster
The 'nnunet_train.sh' script is used to submit a job for training nnU-Net models on an HPC cluster managed with the SLURM workload manager. It runs training across multiple cross-validation folds automatically.

### SLURM Script Explanation
#SBATCH lines: These lines are SLURM directives that specify the job's resource requirements and settings:

--job-name: Sets a name for the job to be submitted.
--account: The SLURM account name used for job accounting.
--time: The wall-clock time limit for the entire job.
--cpus-per-task: The number of CPU cores allocated per task.
--mem: The memory allocated for the job.
--gpus-per-node: The type and number of GPUs allocated per node.
DATASET_ID: A variable representing the unique identifier of the dataset being used for training.

for FOLD in 0 1 2 3 4: A loop that iterates over the folds (0 through 4) for cross-validation training.

nnUNetv2_train: The command used to initiate training with nnU-Net. It includes arguments for the dataset ID, the network dimensionality (2d), the fold number, and an option --npz indicating that the data should be saved in .npz format.

echo: Prints messages to the console for user feedback on the script's progress.

### Using the Script
To use this script, ensure you are on the SLURM-managed HPC system with nnU-Net and all its dependencies correctly installed. You submit the script to the SLURM scheduler using the sbatch command:
```bash
sbatch nnunet_train.sh
```
When the job is submitted, SLURM schedules it for execution based on the current workload and cluster configuration. The script automatically runs the nnU-Net training for each of the five folds, utilizing the specified resources.

It is important to modify the SLURM directives at the top of the script to fit the resource requirements and policies of your particular HPC environment.

## Inference Using nnU-Netv2
The 'nnunet_inference.py' script is designed to apply a trained nnUNet model to new datasets or individual images for segmentation. This process is referred to as inference.
### Features of the Script
- Automatically detects and utilizes GPU resources if available and requested.
- Allows inference on a whole dataset or individual images.
- Converts filenames to nnUNet format temporarily to ensure compatibility.
- Utilizes environment variables to locate nnUNet directories, but allows for overrides via command-line arguments.
- Capable of selecting between the best or the final checkpoint for prediction.

### Prerequisites
Before running the script, ensure that you have the following:

- A trained nnUNetv2 model located in a specified directory.
- A set of images or a dataset on which to perform inference.
- Python environment where nnUNetv2 and its dependencies are installed.

### How to Use the Script
1. Setting Environment Variables: If not already set, the script will set the nnUNet_raw, nnUNet_results, and nnUNet_preprocessed environment variables to 'UNDEFINED'. It is essential to set these correctly before running the script, pointing to the respective directories in your nnUNet framework.
2. Script Arguments:
- --path-dataset: Path to the folder containing a dataset for bulk inference.
- --path-images: List of paths to individual images for inference.
- --path-out: Path to the output directory where segmented images will be saved.
- --path-model: Path to the trained model directory.
- --folds: Specific folds of the model to use for inference.
- --use-gpu: Flag to enable GPU usage for inference.
- --use-best-checkpoint: Flag to use the best checkpoint for prediction instead of the final one.
3. Running the Script:
```bash
python nnunet_inference.py --path-dataset /path/to/dataset --path-model /path/to/model/directory --path-out /path/to/output --use-gpu
```
The script will output the segmented images to the specified --path-out directory. Additionally, it provides console output about the inference process and the location of the results.

## Trnsforming Inference Image Intensities
The inference process of our nnU-Net model outputs images where the pixel intensities are either 0 or 1, representing different classes. However, for better visualization and compatibility with certain image viewing or processing tools, it may be beneficial to convert these intensities, mapping the value 1 to 255. This transformation makes the output images easier to view, as pixels representing the class of interest will be fully white.

### Using the convert_pixels.py Script
We have provided a Python script, convert_pixels.py, which automates this process. This script takes the path to an input image (or images) with intensities of 0 and 1, and an output path where the transformed image will be saved with intensities of 0 and 255.

### How to Use the Script
To transform the pixel intensities of an image, navigate to the directory containing convert_pixels.py in your terminal or command prompt, and run:
```bash
python convert_pixels.py --input_image_path --output_image_path
```
Replace <input_image_path> with the path to your input image file, and <output_image_path> with the desired path for the output image. If <output_image_path> is a directory, the script will automatically generate an output filename based on the input filename.

### Notes
- This script is designed to work with grayscale images (mode 'L' in PIL). If your input images are in a different mode, you may need to modify the script accordingly.
- The script ensures that the output directory exists before saving the image. If the directory does not exist, it will be created.
- If the output path provided is a directory, the script will save the output image in that directory with a modified name indicating it is the final version of the input image.