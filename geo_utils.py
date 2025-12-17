from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import re
import json
import time
import difflib

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
# Helpers (normalisation / similarité)
# ---------------------------------------------------------------------------

def _normalize_text_for_compare(text: str) -> str:
    """
    Normalisation "anti micro-variations" :
    - trim
    - \r\n -> \n
    - collapse espaces
    - collapse lignes vides
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    t = "\n".join(" ".join(line.split()) for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def compute_similarity(original_text: str, final_text: str) -> float:
    """Similarité (0..1) basée sur difflib.SequenceMatcher, après normalisation."""
    a = _normalize_text_for_compare(original_text)
    b = _normalize_text_for_compare(final_text)
    if not a and not b:
        return 1.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


# ---------------------------------------------------------------------------
# GEO "Texte déjà optimisé" (pré-check heuristique binaire et conservateur)
# ---------------------------------------------------------------------------

def geo_is_text_already_optimized(original_text: str, target_query: str) -> bool:
    """
    Pré-check heuristique BINAIRE et CONSERVATEUR.
    True UNIQUEMENT si on est sûr que le texte est déjà GEO-friendly.
    Au moindre doute -> False (fail-open).

    Critères critiques (tous doivent passer) :
    - texte >= 80 mots
    - >= 2 paragraphes
    - lisibilité : moyenne <= 28 mots / phrase et max <= 45
    - couverture sémantique tokens requête >= 0.80 (stopwords simples)
    - neutralité : pas de marqueurs marketing (liste) ; pas d'emojis ; pas de "!" répétés
    """
    try:
        text = (original_text or "").strip()
        query = (target_query or "").strip()
        if not text or not query:
            return False

        words = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", text)
        if len(words) < 80:
            return False

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < 2:
            return False

        sentences = [s.strip() for s in re.split(r"[\.!\?]\s+", text) if s.strip()]
        if len(sentences) < 3:
            return False
        sent_lens = [len(re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", s)) for s in sentences]
        avg_len = sum(sent_lens) / max(1, len(sent_lens))
        if avg_len > 28:
            return False
        if max(sent_lens) > 45:
            return False

        stop = {
            "le","la","les","un","une","des","du","de","d","l",
            "et","ou","mais","donc","or","ni","car",
            "à","au","aux","en","dans","sur","sous","avec","sans","pour","par",
            "ce","cet","cette","ces","son","sa","ses","leur","leurs",
            "qui","que","quoi","dont","où",
            "est","sont","été","être","fait","faire",
            "plus","moins","très","trop","aussi","ainsi","comme",
            "the","a","an","and","or","but","to","of","in","on","for","with",
            "is","are","was","were","be","been","being",
        }

        q_tokens_raw = [t.lower() for t in re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", query)]
        q_tokens = [t for t in q_tokens_raw if t not in stop and len(t) >= 3]
        if not q_tokens:
            return False

        text_tokens_raw = [t.lower() for t in re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", text)]
        text_tokens = {t for t in text_tokens_raw if t not in stop and len(t) >= 3}

        q_unique = set(q_tokens)
        coverage = sum(1 for t in q_unique if t in text_tokens) / max(1, len(q_unique))
        if coverage < 0.80:
            return False

        if re.search(r"!!+", text):
            return False

        try:
            if re.search(r"[\U0001F300-\U0001FAFF]", text):
                return False
        except re.error:
            if re.search(r"[🔥🚀💥🎉✅❌⭐️]", text):
                return False

        marketing = (
            "incroyable","exceptionnel","meilleur","ultime","révolutionnaire",
            "gratuit","promo","offre","profitez","garanti","immanquable","top","100%",
            "incontournable","premium","parfait","idéal","must-have",
        )
        lowered = text.lower()
        if any(w in lowered for w in marketing):
            return False

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Gemini (cloud) - clé / modèle / appel centralisé
# ---------------------------------------------------------------------------

def get_gemini_api_key(user_api_key: Optional[str] = None) -> str:
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


@lru_cache(maxsize=16)
def get_gemini_model(model_name: str):
    genai = _lazy_import_genai()
    return genai.GenerativeModel(model_name)


def _extract_retry_seconds(exc: Exception) -> Optional[int]:
    for attr in ("retry_delay", "retry_after", "retry_after_seconds"):
        val = getattr(exc, attr, None)
        if val is None:
            continue
        try:
            if hasattr(val, "total_seconds"):
                s = int(val.total_seconds())
                if s > 0:
                    return s
            s = int(val)
            if s > 0:
                return s
        except Exception:
            pass

    msg = str(exc) or ""
    msg_low = msg.lower()
    patterns = [
        r"retry\s*(?:in|after)\s*(\d+)\s*(?:s|sec|secs|second|seconds)\b",
        r"retry[-_\s]?after[:\s]*(\d+)\b",
        r"retry[_\s]?delay(?:\s*seconds)?[:=\s]+(\d+)\b",
        r"after\s*(\d+)\s*(?:s|sec|secs|second|seconds)\b",
    ]
    for pat in patterns:
        m = re.search(pat, msg_low)
        if m:
            try:
                s = int(m.group(1))
                if s > 0:
                    return s
            except Exception:
                pass

    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers:
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                s = int(ra)
                if s > 0:
                    return s
            except Exception:
                pass

    return None


def call_gemini(
    prompt: str,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    user_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Appel Gemini centralisé.
    Retourne : {"text": str, "cooldown_seconds": int|None, "error": str|None}
    """
    try:
        configure_gemini(api_key=user_api_key)
        model_id = (model_name or DEFAULT_GEMINI_MODEL).strip()
        model = get_gemini_model(model_id)

        # Tentative 1 : si supporté, on demande du JSON. Si échec, on retente sans.
        gen_cfg = {"temperature": float(temperature), "response_mime_type": "application/json"}
        try:
            resp = model.generate_content(prompt, generation_config=gen_cfg)
        except Exception:
            resp = model.generate_content(prompt, generation_config={"temperature": float(temperature)})

        text = (getattr(resp, "text", "") or "").strip()
        return {"text": text, "cooldown_seconds": None, "error": None}
    except Exception as exc:
        msg = str(exc) or "Erreur Gemini inconnue."
        low = msg.lower()
        if "429" in low or "resource_exhausted" in low or "quota" in low:
            cooldown = _extract_retry_seconds(exc) or 30
            return {"text": "", "cooldown_seconds": int(cooldown), "error": msg}
        return {"text": "", "cooldown_seconds": None, "error": msg}


def test_gemini_connection(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[bool, str]:
    try:
        configure_gemini(api_key=api_key)
        model_id = (model_name or DEFAULT_GEMINI_MODEL).strip()
        model = get_gemini_model(model_id)
        resp = model.generate_content("Réponds simplement : 'OK Gemini GEO'.")
        text = getattr(resp, "text", "") or ""
        if "OK Gemini GEO" in text:
            return True, "OK Gemini GEO"
        return True, f"OK Gemini (réponse: {text[:120]})"
    except Exception as exc:
        return False, f"ERREUR Gemini: {exc}"


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

    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama a renvoyé {response.status_code} : {response.text}")

    data = response.json()
    message = data.get("message", {}) or {}
    return (message.get("content", "") or "").strip()


def test_ollama_connection(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[bool, str]:
    try:
        _ = call_ollama_chat(
            prompt="Réponds simplement : 'OK Ollama GEO'.",
            model=model,
            base_url=base_url,
            temperature=0.0,
            timeout=30,
        )
        return True, "OK Ollama GEO"
    except Exception as exc:
        return False, f"ERREUR Ollama: {exc}"


# ---------------------------------------------------------------------------
# JSON strict prompts + validation
# ---------------------------------------------------------------------------

_JSON_SCHEMA = """{
  "final_text": "string",
  "mode": "minimal|ameliorer|creatif",
  "title_included": false,
  "added_structure": false,
  "added_facts_suspected": false,
  "markdown_suspected": false,
  "notes_for_internal_use": "string"
}"""


def build_prompt_json_strict(
    original_text: str,
    section_title: str,
    rewrite_mode: str,
) -> str:
    src = (original_text or "").strip()
    title = (section_title or "").strip()
    mode = (rewrite_mode or "ameliorer").strip().lower()

    mode_rules = {
        "minimal": (
            "Corrige UNIQUEMENT l'orthographe, la ponctuation, la typographie. "
            "Ne reformule pas. Ne change pas l'ordre. Ne crée aucune structure. "
            "Si aucune correction évidente : renvoie EXACTEMENT le texte source."
        ),
        "ameliorer": (
            "Fluidifie légèrement en restant fidèle. Ne change pas les faits. "
            "Ne change pas l'ordre logique. Ne crée aucune structure (pas de titres, pas de listes)."
        ),
        "creatif": (
            "Propose une version plus différente, MAIS 100% factuelle. "
            "Aucun ajout d'informations non présentes dans la source. "
            "Pas de style romanesque. Aucune structure ajoutée."
        ),
    }
    mode_instruction = mode_rules.get(mode, mode_rules["ameliorer"])

    return f"""
Tu es un éditeur GEO (Generative Engine Optimization).
Tu vas produire une reformulation conforme ET exploitable par des moteurs IA.

CONTEXTE (NE PAS REPRODUIRE) :
- Titre de section / intention (contexte uniquement) : "{title}"

CONTRAINTES ABSOLUES :
- Tu dois retourner UNIQUEMENT un JSON strict, valide, sans markdown, sans ``` et sans texte hors JSON.
- Le JSON doit respecter EXACTEMENT ce schéma (mêmes clés) :
{_JSON_SCHEMA}

RÈGLES SUR final_text :
- TEXTE BRUT uniquement (paragraphes). Pas de sous-titres, pas de sections, pas de listes, pas de puces, pas de numérotation.
- Interdits : tout caractère '#' ; lignes commençant par '- ', '* ', '•' ; lignes commençant par '1.' '2.' etc.
- Interdits : '**', '```', liens markdown.
- Le titre de section NE DOIT PAS être ajouté au début, ni apparaître en première ligne, ni être répété en tête.
- NE JAMAIS inventer de faits, chiffres, dates, citations, sources.

MODE DEMANDÉ : {mode}
RÈGLE DE MODE : {mode_instruction}

TEXTE SOURCE :
{src}
""".strip()


def _looks_like_markdown(text: str) -> bool:
    if not text:
        return False
    if "```" in text or "**" in text:
        return True
    if re.search(r"\[[^\]]+\]\([^)]+\)", text):
        return True
    return False


def validate_output(final_text: str, title: str) -> Tuple[List[str], Dict[str, bool]]:
    violations: List[str] = []
    flags = {
        "title_included": False,
        "added_structure": False,
        "added_facts_suspected": False,
        "markdown_suspected": False,
    }

    txt = (final_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not txt:
        violations.append("final_text_empty")

    t = (title or "").strip()
    if t:
        first_nonempty = ""
        for ln in txt.splitlines():
            if ln.strip():
                first_nonempty = ln.strip()
                break
        if first_nonempty and first_nonempty.casefold().startswith(t.casefold()):
            violations.append("title_included_at_start")
            flags["title_included"] = True
        if txt.casefold().startswith(t.casefold()):
            violations.append("title_included_prefix")
            flags["title_included"] = True

    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            violations.append("heading_detected")
            flags["added_structure"] = True
            flags["markdown_suspected"] = True
            break
        if s.startswith(("- ", "* ", "• ")):
            violations.append("bullet_list_detected")
            flags["added_structure"] = True
            break
        if re.match(r"^\d+[\.|\)]\s+", s):
            violations.append("numbered_list_detected")
            flags["added_structure"] = True
            break

    if _looks_like_markdown(txt):
        violations.append("markdown_detected")
        flags["markdown_suspected"] = True

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if lines:
        short_lines = sum(1 for ln in lines if len(ln) <= 35)
        colon_lines = sum(1 for ln in lines if ln.endswith(":"))
        if short_lines >= 4 or colon_lines >= 2:
            flags["added_structure"] = True
            violations.append("structure_suspected")

    return list(dict.fromkeys(violations)), flags


def _parse_json_strict(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if raw is None:
        return None, "json_missing"
    s = raw.strip()
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None, "json_not_object"
        return obj, None
    except Exception:
        return None, "json_invalid"


def _clean_text_locally(text: str) -> str:
    if not text:
        return ""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)
    cleaned_lines = []
    for ln in t.splitlines():
        s = ln.rstrip()
        if re.match(r"^\s*#{1,6}\s+", s):
            continue
        if re.match(r"^\s*(\-|\*|•)\s+", s):
            continue
        if re.match(r"^\s*\d+[\.|\)]\s+", s):
            continue
        cleaned_lines.append(s)
    t = "\n".join(cleaned_lines)
    t = t.replace("**", "")
    return _normalize_text_for_compare(t)


def repair_output(
    original_text: str,
    bad_text: str,
    section_title: str,
    rewrite_mode: str,
    backend: str = "gemini",
    model_name: Optional[str] = None,
    user_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    src = (original_text or "").strip()
    bad = (bad_text or "").strip()
    title = (section_title or "").strip()
    mode = (rewrite_mode or "ameliorer").strip().lower()

    prompt = f"""
Tu es un correcteur de sortie. Tu dois réparer une sortie non conforme.
Tu dois retourner UNIQUEMENT un JSON strict valide conforme au schéma :
{_JSON_SCHEMA}

CONTRAINTES ABSOLUES sur final_text :
- TEXTE BRUT uniquement (paragraphes). Pas de titres, pas de sections, pas de listes, pas de puces, pas de numérotation, pas de markdown.
- Interdits : '#', '- ', '* ', '•', '1.' etc, '**', '```'.
- Le titre de section NE DOIT PAS apparaître en première ligne ni être répété en tête.
- Aucun ajout de faits : ne rajoute aucune info.

CONTEXTE (NE PAS REPRODUIRE) : "{title}"
MODE : {mode}

TEXTE SOURCE :
{src}

SORTIE NON CONFORME À RÉPARER :
{bad}
""".strip()

    backend_norm = (backend or "gemini").strip().lower()
    if backend_norm == "ollama":
        raw = call_ollama_chat(prompt)
        return {"raw": raw, "cooldown_seconds": None, "error": None}

    call = call_gemini(
        prompt=prompt,
        model_name=model_name or DEFAULT_GEMINI_MODEL,
        temperature=0.0,
        user_api_key=user_api_key,
    )
    return {"raw": call.get("text", ""), "cooldown_seconds": call.get("cooldown_seconds"), "error": call.get("error")}


def geo_rewrite_content(
    original_text: str,
    target_query: str,
    model_name: Optional[str] = None,
    rewrite_mode: str = "ameliorer",
    backend: str = "gemini",
    user_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "text": "",
        "already_optimized": False,
        "similarity": 0.0,
        "repaired": False,
        "violations": [],
        "cooldown_seconds": None,
        "error": None,
    }

    if original_text and len(original_text) > 10000:
        result["text"] = (original_text or "").strip()
        result["error"] = (
            "Le texte est trop long pour une reformulation en une seule fois (max 10k caractères). "
            "Découpe-le en sections."
        )
        return result

    src = (original_text or "").strip()
    title = (target_query or "").strip()
    mode = (rewrite_mode or "ameliorer").strip().lower()
    backend_norm = (backend or "gemini").strip().lower()
    model_id = (model_name or DEFAULT_GEMINI_MODEL).strip()

    if not src:
        result["error"] = "Texte original vide."
        return result
    if not title:
        result["error"] = "Titre de section / requête cible vide."
        return result

    prompt = build_prompt_json_strict(src, title, mode)

    raw = ""
    if backend_norm == "ollama":
        try:
            raw = call_ollama_chat(prompt)
        except Exception as exc:
            result["text"] = src
            result["error"] = str(exc)
            return result
    else:
        call = call_gemini(prompt=prompt, model_name=model_id, temperature=0.2, user_api_key=user_api_key)
        raw = (call.get("text", "") or "").strip()
        if call.get("cooldown_seconds"):
            result["cooldown_seconds"] = int(call["cooldown_seconds"])
        if call.get("error"):
            result["text"] = src
            result["error"] = call["error"]
            return result

    obj, parse_err = _parse_json_strict(raw)
    if obj is None:
        result["violations"] = [parse_err or "json_invalid"]
        rep = repair_output(src, raw, title, mode, backend=backend_norm, model_name=model_id, user_api_key=user_api_key)
        if rep.get("cooldown_seconds"):
            result["cooldown_seconds"] = int(rep["cooldown_seconds"])
        if rep.get("error"):
            result["text"] = src
            result["error"] = rep["error"]
            return result
        obj2, parse_err2 = _parse_json_strict(rep.get("raw", ""))
        if obj2 is None:
            result["text"] = src if mode == "minimal" else (_clean_text_locally(raw) or src)
            result["error"] = "Sortie non conforme (JSON invalide). Relecture requise."
            result["similarity"] = compute_similarity(src, result["text"])
            return result
        obj = obj2
        result["repaired"] = True

    final_text = (obj.get("final_text") or "").strip()
    out_mode = (obj.get("mode") or mode).strip().lower()
    if out_mode not in ("minimal", "ameliorer", "creatif"):
        out_mode = mode

    violations, _flags = validate_output(final_text, title)

    if violations:
        result["violations"] = violations
        rep = repair_output(src, final_text, title, out_mode, backend=backend_norm, model_name=model_id, user_api_key=user_api_key)
        if rep.get("cooldown_seconds"):
            result["cooldown_seconds"] = int(rep["cooldown_seconds"])
        if rep.get("error"):
            result["text"] = src
            result["error"] = rep["error"]
            return result

        obj2, parse_err2 = _parse_json_strict(rep.get("raw", ""))
        if obj2 is None:
            result["text"] = src if out_mode == "minimal" else (_clean_text_locally(final_text) or src)
            result["error"] = "Sortie non conforme (repair JSON invalide). Relecture requise."
            result["similarity"] = compute_similarity(src, result["text"])
            return result

        repaired_text = (obj2.get("final_text") or "").strip()
        violations2, _ = validate_output(repaired_text, title)
        if violations2:
            if out_mode == "minimal":
                result["text"] = src
            else:
                cleaned = _clean_text_locally(repaired_text)
                v3, _ = validate_output(cleaned, title)
                result["text"] = cleaned if cleaned and not v3 else src
            result["repaired"] = True
            result["violations"] = list(dict.fromkeys(result["violations"] + violations2))
            result["error"] = "Sortie non conforme après repair. Relecture requise."
            result["similarity"] = compute_similarity(src, result["text"])
            return result

        final_text = repaired_text
        result["repaired"] = True

    sim = compute_similarity(src, final_text)
    result["similarity"] = sim

    orig_norm = _normalize_text_for_compare(src)
    final_norm = _normalize_text_for_compare(final_text)

    already_optimized = (orig_norm == final_norm)
    if out_mode == "minimal" and sim >= 0.985:
        already_optimized = True
        final_text = src

    result["already_optimized"] = bool(already_optimized)
    result["text"] = final_text
    return result


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
