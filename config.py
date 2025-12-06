import os
from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env (si présent)
load_dotenv()

# Environnement global (dev / prod)
# GEO_ENV permet de distinguer l’usage local (dev) de la version hébergée (prod).
# En local, laisse GEO_ENV=dev. Sur le serveur, définis GEO_ENV=prod.
GEO_ENV = os.getenv("GEO_ENV", "dev").lower()
IS_PROD = GEO_ENV == "prod"

# Backend LLM par défaut : "ollama" ou "gemini"
DEFAULT_BACKEND = os.getenv("DEFAULT_BACKEND", "gemini").lower()

# Configuration Gemini
# Configuration Gemini

GEMINI_MODEL_REFORMULATION = "gemini-2.5-flash"
GEMINI_MODEL_MONITORING = "gemini-2.5-flash-lite"

DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")

# Configuration Ollama (LLM local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Modèle par défaut pour Ollama (penser à exécuter : `ollama pull qwen2.5:7b`)
# Variante plus lourde possible : `qwen2.5:14b` (après `ollama pull qwen2.5:14b`)
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")



