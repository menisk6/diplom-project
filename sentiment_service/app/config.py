from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_STORE_DIR = BASE_DIR / 'model_store'
CACHE_DIR = BASE_DIR / 'cache'

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = MODEL_STORE_DIR / 'registry.json'
