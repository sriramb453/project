import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def save_artifact(artifact, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(artifact, filepath)
    print(f"Successfully saved to {filepath}")

def load_artifact(filename):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)