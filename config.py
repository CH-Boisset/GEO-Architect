import os

from dotenv import load_dotenv

load_dotenv()


def get_env_var(names, default=None):
    """
    Récupère une variable de configuration depuis les variables d'environnement.

    `names` peut être une chaîne ou une liste de noms possibles.
    On retourne la première valeur non vide trouvée, sinon `default`.
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value

    return default


GEO_ENV = (get_env_var("GEO_ENV", default="prod") or "prod").lower()
IS_PROD = GEO_ENV == "prod"

DEFAULT_GEMINI_MODEL = get_env_var(
    "DEFAULT_GEMINI_MODEL",
    default="gemini-2.5-flash",
)

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
