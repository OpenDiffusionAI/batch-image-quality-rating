import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse
import os
from PIL import Image
import os
import sqlite3

parser = argparse.ArgumentParser(description='Image Quality Assessment')
parser.add_argument('--db', type=str, help='Path to an SQLite database file')
parser.add_argument('--image', type=str, help='Path to a single image')
parser.add_argument('--dir', type=str, help='Path to a folder of images')
parser.add_argument('--r', action='store_true', help='Recursively search for images in the directory provided')
parser.add_argument('--q4', action='store_true', help='Load model with 4-bit quantization (~4GB + PyTorch)')
parser.add_argument('--q8', action='store_true', help='Load model with 8-bit quantization (~8GB + PyTorch)')
args = parser.parse_args()

if args.db:
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    
    cursor.execute("CREATE TABLE IF NOT EXISTS ratings (path TEXT PRIMARY KEY UNIQUE, quality NUMERIC(9, 8), aesthetic NUMERIC(9, 8), inf_precision INTEGER(2), timestamp NUMERIC(24))")
else:
    conn = None
    cursor = None

inf_precision = 8 if args.q8 else 4 if args.q4 else 16

def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
        if not args.r:
            break
    return image_paths

if args.dir:
    images = get_image_paths(args.dir)
elif args.image:
    if args.image.lower().endswith(('.jpg', '.png', '.jpeg')):
        images = [args.image]
    else:
        print("Invalid image file format. Please provide an image file ending in .jpg, .png, or .jpeg.")
        exit()
else:
    print("Please provide either --image or --dir argument.")
    exit()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "q-future/one-align", 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    load_in_4bit=args.q4, 
    load_in_8bit=args.q8,
    quantization_config=bnb_config if args.q4 else None,
)

def score_images():
    for img in images:
        
        startTime = time.time()
        
        try:
            
            if cursor is not None:
                if cursor.execute("SELECT * FROM ratings WHERE path = ? AND inf_precision >= ?", (img, inf_precision)).fetchone():
                    print("Image already rated at this precision, skipping...")
                    continue
            
            imageData = Image.open(img)
            
            if imageData.mode != "RGB":
                imageData = imageData.convert("RGB")
                
            startTime = time.time()
            
            qualityScore = model.score([imageData], task_="quality", input_="image")
            aestheticScore = model.score([imageData], task_="aesthetic", input_="image")
            
            if cursor is not None:
                if cursor.execute("SELECT * FROM ratings WHERE path = ?", (img,)).fetchone():
                    cursor.execute("UPDATE ratings SET quality = ?, aesthetic = ?, inf_precision = ?, timestamp = ? WHERE path = ?", (qualityScore.item(), aestheticScore.item(), inf_precision, int(time.time()), img))
                else:
                    cursor.execute("INSERT INTO ratings (path, quality, aesthetic, inf_precision, timestamp) VALUES (?, ?, ?, ?, ?)", (img, qualityScore.item(), aestheticScore.item(), inf_precision, int(time.time())))
                
                conn.commit()
                
        except Exception as e:
            print(e)
            continue
        
        responseObject = { "quality": f"{qualityScore.item():.4f}", "aesthetic": f"{aestheticScore.item():.4f}", "time": f"{time.time() - startTime:.2f} seconds"}
        print(responseObject)

score_images()

if conn is not None:
    conn.close()

print("Inferencing completed. Shutting down...")

# The last release where this works is torch 2.1.2:
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu121 --upgrade