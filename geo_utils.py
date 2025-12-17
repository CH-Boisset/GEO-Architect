from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional
import urllib.parse
import re

import requests

from config import (
    get_env_var,
    DEFAULT_GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# Lazy imports (évite que l'app crashe au chargement si une lib manque)
# ---------------------------------------------------------------------------

def _lazy_import_genai():
    try:
        import google.generativeai as genai  # type: ignore
        return genai
    except Exception as exc:
        raise RuntimeError(
            "Import google-generativeai impossible. "
            "Vérifie que 'google-generativeai' est bien dans requirements.txt."
        ) from exc


def _lazy_import_bs4():
    try:
        from bs4 import BeautifulSoup  # type: ignore
        return BeautifulSoup
    except Exception as exc:
        raise RuntimeError(
            "Import BeautifulSoup impossible. "
            "Vérifie que 'beautifulsoup4' est bien dans requirements.txt."
        ) from exc


def _lazy_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as exc:
        raise RuntimeError(
            "Import pandas impossible. "
            "Vérifie que 'pandas' est bien dans requirements.txt."
        ) from exc


# ---------------------------------------------------------------------------
# GEO Heuristics (Added Hotfix)
# ---------------------------------------------------------------------------

def geo_is_text_already_optimized(original_text: str, target_query: str) -> bool:
    """
    Détection heuristique BINAIRE et CONSERVATRICE :
    - True UNIQUEMENT si l’on est très confiant que le texte est déjà GEO-friendly.
    - Au moindre doute : False  => on reformule.
    """
    try:
        text = (original_text or "").strip()
        query = (target_query or "").strip()
        if not text or not query:
            return False

        # 1) Texte suffisamment long (seuil conservateur)
        words = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", text)
        if len(words) < 80:
            return False

        # 2) Structure : >= 2 paragraphes + (titre OU liste)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < 2:
            return False

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        has_heading = any(
            ln.startswith("#") or (ln.isupper() and 8 <= len(ln) <= 80)
            for ln in lines
        )
        has_bullets = any(re.match(r"^(\-|\*|•|\d+[\.|\)])\s+", ln) for ln in lines)
        if not (has_heading or has_bullets):
            return False

        # 3) Lisibilité : éviter phrases trop longues
        sentences = [s.strip() for s in re.split(r"[\.\!\?]\s+", text) if s.strip()]
        if len(sentences) < 3:
            return False
        sent_lens = [len(re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", s)) for s in sentences]
        avg_len = sum(sent_lens) / max(1, len(sent_lens))
        if avg_len > 28:
            return False
        if max(sent_lens) > 45:
            return False

        # 4) Couverture sémantique : présence forte des tokens de requête
        stop = {
            "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
            "et", "ou", "mais", "donc", "or", "ni", "car",
            "à", "au", "aux", "en", "dans", "sur", "sous", "avec", "sans", "pour", "par",
            "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs",
            "qui", "que", "quoi", "dont", "où",
            "est", "sont", "été", "être", "fait", "faire",
            "plus", "moins", "très", "trop", "aussi", "ainsi", "comme",
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
            "is", "are", "was", "were", "be", "been", "being",
        }

        q_tokens_raw = [t.lower() for t in re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", query)]
        q_tokens = [t for t in q_tokens_raw if t not in stop and len(t) >= 3]
        if not q_tokens:
            return False

        text_tokens_raw = [t.lower() for t in re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", text)]
        text_tokens = {t for t in text_tokens_raw if t not in stop and len(t) >= 3}

        coverage = sum(1 for t in set(q_tokens) if t in text_tokens) / max(1, len(set(q_tokens)))
        if coverage < 0.80:
            return False

        # 5) Neutralité : refuser marqueurs marketing évidents
        if "!" in text:
            return False
        if re.search(r"[🔥🚀💥🎉]", text):
            return False
        marketing = (
            "incroyable", "exceptionnel", "meilleur", "ultime", "révolutionnaire",
            "gratuit", "promo", "offre", "profitez", "garanti", "immanquable", "top", "100%"
        )
        lowered = text.lower()
        if any(w in lowered for w in marketing):
            return False

        return True

    except Exception:
        # En cas d'incident : on ne bloque pas => reformulation
        return False


# ---------------------------------------------------------------------------
# Gemini (cloud)
# ---------------------------------------------------------------------------

def get_gemini_api_key(user_api_key: Optional[str] = None) -> str:
    """
    Récupère la clé Gemini :
    - user_api_key (si fourni)
    - env/secrets : GEMINI_API_KEY ou GOOGLE_API_KEY
    """
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()

    key = get_env_var(["GEMINI_API_KEY", "GOOGLE_API_KEY"], default="")
    key = (key or "").strip()
    if not key:
        raise RuntimeError(
            "Aucune clé API Gemini trouvée. "
            "Définis GEMINI_API_KEY (ou GOOGLE_API_KEY) dans l'environnement ou dans Streamlit Secrets."
        )
    return key


def configure_gemini(api_key: Optional[str] = None) -> None:
    genai = _lazy_import_genai()
    key = get_gemini_api_key(user_api_key=api_key)
    genai.configure(api_key=key)


@lru_cache(maxsize=8)
def get_gemini_model(model_name: str):
    genai = _lazy_import_genai()
    return genai.GenerativeModel(model_name)


def test_gemini_connection(
    model_name: Optional[str] = None,
    user_api_key: Optional[str] = None,
) -> str:
    """
    Test simple Gemini. Retourne un message texte (utilisé par app.py).
    """
    try:
        configure_gemini(api_key=user_api_key)
        model_id = (model_name or DEFAULT_GEMINI_MODEL).strip()
        model = get_gemini_model(model_id)
        resp = model.generate_content("Réponds simplement : 'OK Gemini GEO'.")
        text = getattr(resp, "text", "") or ""
        if "OK Gemini GEO" in text:
            return "OK Gemini GEO"
        return f"OK Gemini (réponse: {text[:80]})"
    except Exception as exc:
        return f"ERREUR Gemini: {exc}"


def call_gemini_text(prompt: str, model_name: Optional[str] = None, temperature: float = 0.2) -> str:
    configure_gemini()
    model_id = (model_name or DEFAULT_GEMINI_MODEL).strip()
    model = get_gemini_model(model_id)
    resp = model.generate_content(prompt, generation_config={"temperature": float(temperature)})
    return (getattr(resp, "text", "") or "").strip()


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

def call_ollama_chat(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    base = (base_url or OLLAMA_BASE_URL).rstrip("/")
    url = f"{base}/api/chat"

    payload = {
        "model": model or OLLAMA_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Impossible de se connecter à Ollama.\n"
            "- Vérifie que l'application Ollama est lancée.\n"
            f"- Vérifie que l'API répond sur {base}\n"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Erreur lors de l'appel à Ollama : {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Ollama a renvoyé {response.status_code} : {response.text}")

    data = response.json()
    message = data.get("message", {}) or {}
    return (message.get("content", "") or "").strip()


def test_ollama_connection(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Test simple Ollama. Retourne un message texte (utilisé par app.py).
    """
    try:
        _ = call_ollama_chat(
            prompt="Réponds simplement : 'OK Ollama GEO'.",
            model=model,
            base_url=base_url,
            temperature=0.0,
            timeout=30,
        )
        return "OK Ollama GEO"
    except Exception as exc:
        return f"ERREUR Ollama: {exc}"


# ---------------------------------------------------------------------------
# GEO prompt + rewrite
# ---------------------------------------------------------------------------

def build_geo_prompt(original_text: str, target_query: str, rewrite_mode: str = "ameliorer") -> str:
    original_text = (original_text or "").strip()
    target_query = (target_query or "").strip()
    rewrite_mode = (rewrite_mode or "ameliorer").strip().lower()

    mode_rules = {
        "minimal": "Change le moins possible. Corrige surtout la clarté, la structure, et la couverture sémantique.",
        "ameliorer": "Améliore clairement la structure et la couverture sémantique, tout en restant fidèle au fond.",
        "creatif": "Améliore avec un peu plus de fluidité, mais reste factuel et non lyrique.",
    }
    mode_instruction = mode_rules.get(rewrite_mode, mode_rules["ameliorer"])

    instructions = f"""
Tu es un éditeur GEO (Generative Engine Optimization).
Objectif : produire un texte exploitable par des moteurs IA (structure claire, ton neutre, faits).
Contraintes :
- NE PAS inventer de faits, chiffres, dates, citations.
- NE PAS ajouter de détails non présents dans la source.
- Conserver toutes les idées importantes.
- Style : factuel, neutre, lisible, sections courtes, listes si utile.
- Optimiser la couverture sémantique autour de la requête cible : "{target_query}".

Niveau demandé : {rewrite_mode}
Règle de mode : {mode_instruction}
""".strip()

    return f"""{instructions}

--- TEXTE SOURCE ---
{original_text}

--- TEXTE REFORMULÉ (sortie finale uniquement) ---
"""


def geo_rewrite_content(
    original_text: str,
    target_query: str,
    model_name: Optional[str] = None,
    rewrite_mode: str = "ameliorer",
    backend: str = "ollama",
    user_api_key: Optional[str] = None,
) -> str:
    if original_text and len(original_text) > 10000:
        raise ValueError(
            "Le texte est trop long pour une reformulation en une seule fois (max 10k caractères). "
            "Découpe-le en sections."
        )

    original_text = (original_text or "").strip()
    target_query = (target_query or "").strip()
    if not original_text:
        raise ValueError("Texte original vide.")
    if not target_query:
        raise ValueError("Requête cible vide.")

    prompt = build_geo_prompt(original_text, target_query, rewrite_mode=rewrite_mode)
    backend = (backend or "gemini").strip().lower()

    if backend == "ollama":
        return call_ollama_chat(prompt)

    configure_gemini(api_key=user_api_key)
    return call_gemini_text(prompt, model_name=model_name or DEFAULT_GEMINI_MODEL, temperature=0.2)


# ---------------------------------------------------------------------------
# Monitoring (DuckDuckGo HTML)
# ---------------------------------------------------------------------------

@dataclass
class MonitoringResult:
    query: str
    rank: int
    title: str
    url: str
    snippet: str
    brand_present: bool


DDG_SEARCH_URL = "https://duckduckgo.com/html/"


def _clean_ddg_url(raw_href: str) -> str:
    if not raw_href:
        return ""
    href = raw_href.strip()
    if href.startswith("/l/") or "uddg=" in href:
        try:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return urllib.parse.unquote(qs["uddg"][0])
        except Exception:
            return href
    return href


def _fetch_duckduckgo_results(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    BeautifulSoup = _lazy_import_bs4()

    params = {"q": query, "kl": "fr-fr"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0 Safari/537.36"
        )
    }

    r = requests.get(DDG_SEARCH_URL, params=params, headers=headers, timeout=20)
    if r.status_code >= 400:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    results: List[Dict[str, str]] = []

    for res in soup.select("div.result"):
        link = res.select_one("a.result__a")
        snippet_el = res.select_one("a.result__snippet") or res.select_one("div.result__snippet")
        if not link:
            continue

        title = link.get_text(" ", strip=True)
        raw_href = link.get("href", "")
        final_url = _clean_ddg_url(raw_href)
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        results.append({"title": title, "url": final_url, "snippet": snippet})
        if len(results) >= int(max_results):
            break

    return results


def monitor_keywords(
    queries: List[str],
    brand_or_domain: str,
    max_results: int = 10,
):
    """
    Retourne un DataFrame (utilisé par app.py).
    Lazy import pandas pour éviter de crasher l'app au démarrage.
    """
    pd = _lazy_import_pandas()

    brand = (brand_or_domain or "").strip().lower()
    rows: List[MonitoringResult] = []

    for query in queries:
        q = (query or "").strip()
        if not q:
            continue

        raw_results = _fetch_duckduckgo_results(q, max_results=max_results)
        for idx, item in enumerate(raw_results, start=1):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            concat = f"{title} {url} {snippet}".lower()
            brand_present = brand in concat

            rows.append(
                MonitoringResult(
                    query=q,
                    rank=idx,
                    title=title,
                    url=url,
                    snippet=snippet,
                    brand_present=brand_present,
                )
            )

    return pd.DataFrame([dataclasses.asdict(r) for r in rows])
