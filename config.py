import os

from dotenv import load_dotenv

# Charge les variables depuis un éventuel fichier .env (en local)
load_dotenv()


def get_env_var(names, default=None):
    """
    Récupère une variable de configuration depuis les variables d'environnement.

    `names` peut être :
    - une chaîne (nom unique),
    - ou une liste de noms possibles.

    On retourne la première valeur non vide trouvée, sinon `default`.
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value

    return default


# ---------------------------------------------------------------------------
# ENVIRONNEMENT
# ---------------------------------------------------------------------------

# En production (Streamlit Cloud), on veut "prod" par défaut.
# En local, l'utilisateur peut forcer GEO_ENV="dev" dans .env.
GEO_ENV = (get_env_var("GEO_ENV", default="prod") or "prod").lower()
IS_PROD = GEO_ENV == "prod"

# Backend par défaut :
# - En prod : on forcera de toute façon Gemini dans l'UI.
# - En dev : permet de choisir le backend initial (Gemini ou Ollama).
DEFAULT_BACKEND = (get_env_var("DEFAULT_BACKEND", default="gemini") or "gemini").lower()

# Modèle Gemini par défaut (texte)
DEFAULT_GEMINI_MODEL = get_env_var(
    "DEFAULT_GEMINI_MODEL",
    default="gemini-2.5-flash",
)

# Modèles Gemini dédiés (si besoin d'affiner par usage)
GEMINI_MODEL_REFORMULATION = "gemini-2.5-flash"
GEMINI_MODEL_MONITORING = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# CONFIG OLLAMA (LLM local pour le dev)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = get_env_var("OLLAMA_BASE_URL", default="http://localhost:11434")
OLLAMA_MODEL_NAME = get_env_var("OLLAMA_MODEL_NAME", default="qwen2.5:7b")
