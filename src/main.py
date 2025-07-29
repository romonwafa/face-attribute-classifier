import sys
import os
import shutil
import logging
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPImageProcessor
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from src.models.multitask_clip_vision_model import MultiTaskClipVisionModel

from src.config import paths

# --- Setup logging ---
os.makedirs(paths.LOG_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(paths.LOG_FILE_PATH, mode='w'),
        logging.StreamHandler()
    ]
)

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "syntheticbot/clip-face-attribute-classifier"
logging.info(f"Using device: {DEVICE}")

AGE_LABELS = sorted(['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70'])
GENDER_LABELS = sorted(['Female', 'Male'])
RACE_LABELS = sorted(['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White'])

ID_MAPPINGS = {
    'age': {i: label for i, label in enumerate(AGE_LABELS)},
    'gender': {i: label for i, label in enumerate(GENDER_LABELS)},
    'race': {i: label for i, label in enumerate(RACE_LABELS)},
}
NUM_LABELS = {
    'age': len(AGE_LABELS),
    'gender': len(GENDER_LABELS),
    'race': len(RACE_LABELS),
}


def classify_images():
    os.makedirs(paths.RESULTS_FOLDER, exist_ok=True)

    # Load face attribute model and processor
    try:
        logging.info(f"Downloading model: {MODEL_ID}")
        model_path = snapshot_download(repo_id=MODEL_ID)
        face_processor = CLIPImageProcessor.from_pretrained(model_path)

        model = MultiTaskClipVisionModel(num_labels=NUM_LABELS)
        weights_path = os.path.join(model_path, "model.safetensors")
        state_dict = load_file(weights_path, device=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()

    except Exception as e:
        logging.error(f"Failed to load model or processor: {e}")
        logging.error("Ensure required libraries and model files are available.")
        return

    if not os.path.exists(paths.INPUT_FOLDER):
        logging.error(f"Input folder not found: {paths.INPUT_FOLDER}")
        return

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    for filename in os.listdir(paths.INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue

        input_path = os.path.join(paths.INPUT_FOLDER, filename)
        logging.info(f"Processing: {filename}")

        try:
            image = Image.open(input_path).convert("RGB")
            inputs = face_processor(images=image, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                logits = model(pixel_values=inputs["pixel_values"])

            predictions = {
                task: ID_MAPPINGS[task][torch.argmax(logits[task]).item()]
                for task in ["age", "gender", "race"]
            }

            gender = predictions["gender"]
            race = predictions["race"]
            age = predictions["age"]

            target_dir = os.path.join(paths.RESULTS_FOLDER, gender, race, age)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy2(input_path, os.path.join(target_dir, filename))
            logging.info(f"Saved to: {target_dir}")
            stats["processed"] += 1

        except UnidentifiedImageError:
            logging.warning(f"Skipped: {filename} — not a valid image.")
            stats["skipped"] += 1
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            stats["errors"] += 1

    logging.info("--- Done ---")
    for k, v in stats.items():
        logging.info(f"{k.capitalize()}: {v}")


if __name__ == "__main__":
    classify_images()
