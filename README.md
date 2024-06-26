# batch-image-quality-rating
Simple python script to rate image quality and aesthetics using one-align

## Requirements
This torch version worked for me with an NVIDIA card

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu121 --upgrade

I recommend setting up a venv

## Dependencies
- torch
- transformers
- pil

## Usage
python inference.py [--db path] [--image path] [--dir path] [--r] [--q4 | --q8]

### Arguments
- --db {path} : Path to an SQLite database file | Optional - To store results
- --image {path} : Path to a single image
- --dir {path} : Path to a folder of images
- --r : Recursively search for images in the directory provided | Optional
- --q4 : Load model with 4-bit quantization (~4GB + PyTorch) | Optional
- --q8 : Load model with 8-bit quantization (~8GB + PyTorch) | Optional
