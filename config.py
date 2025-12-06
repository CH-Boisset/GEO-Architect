import os

from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env (si présent)
load_dotenv()

try:
    import streamlit as st
    _SECRETS = dict(st.secrets)
except Exception:
    _SECRETS = {}

def get_env_var(names, default=None):
    """
    Récupère une variable de configuration depuis :
    1. os.environ (variables d'environnement classiques)
    2. st.secrets (si disponible, ex. sur Streamlit Cloud)

    names peut être une chaîne ou une liste de noms possibles.
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        # 1) Variables d'environnement classiques
        value = os.getenv(name)
        if value not in (None, ""):
            return value

        # 2) Secrets Streamlit (si présents)
        if _SECRETS and name in _SECRETS and _SECRETS[name] not in ("", None):
            return str(_SECRETS[name])

    return default

# En prod (Streamlit Cloud), on veut que l'app se comporte comme "prod" par défaut.
# En local, tu peux forcer GEO_ENV=dev dans un .env pour retrouver ton mode de test.
GEO_ENV = (get_env_var("GEO_ENV", default="prod") or "prod").lower()
IS_PROD = GEO_ENV == "prod"

# Modèle texte par défaut pour Gemini (backbone de GEO Architect)
DEFAULT_GEMINI_MODEL = get_env_var(
    "DEFAULT_GEMINI_MODEL",
    default="gemini-2.5-flash",
)

# Backend par défaut : Gemini (en prod comme en déploiement standard)
DEFAULT_BACKEND = get_env_var(
    "DEFAULT_BACKEND",
    default="gemini",
)

# Configuration Gemini (constantes internes si besoin, mais on privilégie DEFAULT_GEMINI_MODEL)
GEMINI_MODEL_REFORMULATION = "gemini-2.5-flash"
GEMINI_MODEL_MONITORING = "gemini-2.5-flash-lite"

# Configuration Ollama (LLM local)
OLLAMA_BASE_URL = get_env_var("OLLAMA_BASE_URL", default="http://localhost:11434")
OLLAMA_MODEL_NAME = get_env_var("OLLAMA_MODEL_NAME", default="qwen2.5:7b")



