# batch-image-quality-rating
Simple python script to rate image quality and aesthetics using one-align

## Requirements
### Hardware
The unquantized version of this model takes about 16GB of VRAM, luckily, JIT quantization doesn't seem to cause much quality loss.
- 3090 - 4090: Can run full unquantized
- 3080+ : Can run in 8-bit
- 2070+ : Can run in 4-bit

### Torch version
This torch version worked for me with an NVIDIA card. I had issues with the latest version.
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

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

## Behavior
Prints ratings to console as they are generated

Stores ratings in SQLite database if path provided
- Will create new database if one does not exist in that location

![image](https://github.com/OpenDiffusionAI/batch-image-quality-rating/assets/172853169/0549f3d6-854b-4277-a43f-1828e900ab29)
