import os
import shutil

from constants import REPOSITORY_PATH
from factory import load_tbert_encoder

if __name__ == "__main__":
    model = load_tbert_encoder()
    model_path = os.path.join(REPOSITORY_PATH, "tbert_encoder")
    shutil.rmtree(model_path, ignore_errors=True)
    print("Model has been removed.")
    model.save_pretrained(model_path)
    print("New model has been saved.")
