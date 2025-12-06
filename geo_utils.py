from __future__ import annotations

import dataclasses
from typing import List, Optional, Dict, Any

import requests
import pandas as pd
from bs4 import BeautifulSoup
import google.generativeai as genai

from config import (
    DEFAULT_GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
)


# =========================
# LLM - GEMINI (CLOUD)
# =========================

import os

def get_gemini_api_key() -> Optional[str]:
    """
    Récupère la clé API Gemini depuis la variable d'environnement GEMINI_API_KEY.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def configure_gemini(api_key: Optional[str] = None) -> None:
    """
    Configure la librairie google-generativeai avec la clé API fournie
    ou celle présente dans les variables d'environnement.
    """
    key = (api_key or get_gemini_api_key() or "").strip()
    if not key:
        raise RuntimeError(
            "Aucune clé API Gemini trouvée. "
            "Définis GEMINI_API_KEY (ou GOOGLE_API_KEY) dans ton environnement "
            "ou fournis une clé via l'interface."
        )

    try:
        genai.configure(api_key=key)
    except Exception as exc:
        raise RuntimeError(f"Erreur lors de la configuration de Gemini : {exc}") from exc


def get_gemini_model(model_name: str) -> genai.GenerativeModel:
    """
    Instancie un modèle Gemini à partir de son nom.
    """
    try:
        return genai.GenerativeModel(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible d'instancier le modèle Gemini '{model_name}' : {exc}"
        ) from exc


# =========================
# LLM - OLLAMA (LOCAL)
# =========================

def call_ollama_chat(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    timeout: int = 120,
) -> str:
    """
    Appelle l'API locale d'Ollama (/api/chat) avec un prompt texte unique.

    - prompt : texte complet (instructions + contenu à reformuler)
    - model : nom du modèle Ollama (par défaut OLLAMA_MODEL_NAME)
    - base_url : URL de base (par défaut OLLAMA_BASE_URL)
    - temperature : niveau de créativité
    """
    model_name = model or OLLAMA_MODEL_NAME
    base = (base_url or OLLAMA_BASE_URL).rstrip("/")
    url = f"{base}/api/chat"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Impossible de se connecter à Ollama.\n"
            "- Vérifie que l'application Ollama est lancée.\n"
            "- Vérifie que l'API répond (curl http://localhost:11434/api/tags).\n"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Erreur lors de l'appel à Ollama : {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama a renvoyé un statut HTTP {response.status_code} : {response.text}"
        )

    try:
        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
    except Exception as exc:
        raise RuntimeError(
            f"Réponse inattendue d'Ollama (JSON invalide) : {exc}"
        ) from exc

    if not content:
        raise RuntimeError("Réponse vide d'Ollama (champ 'message.content' manquant).")

    return content.strip()


def test_ollama_connection(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Envoie une requête de test à Ollama pour vérifier que :
    - l'API est accessible
    - le modèle demandé peut répondre.
    """
    try:
        _ = call_ollama_chat(
            prompt="Réponds simplement : 'OK Ollama GEO'.",
            model=model,
            base_url=base_url,
            temperature=0.0,
            timeout=30,
        )
        return "Connexion à Ollama OK ✅"
    except Exception as exc:
        return f"Connexion à Ollama KO ❌ : {exc}"


# =========================
# LOGIQUE GEO COMMUNE
# =========================


GEO_SYSTEM_INSTRUCTIONS = """
Tu es un assistant spécialisé en reformulation de contenu pour le GEO (Generative Engine Optimization) en français.

Rôle :
- Réécrire un texte fourni en français en l'optimisant pour les moteurs de recherche (web et IA) tout en restant totalement fidèle au contenu d'origine.
- Servir aussi de correcteur : tu dois corriger toutes les fautes d'orthographe, de grammaire, de typographie et de ponctuation.

Règles générales :
- Langue : français uniquement, sans anglicismes inutiles.
- Orthographe et grammaire : impeccables, aucune faute n'est tolérée.
- Sens et faits : tu dois respecter strictement la chronologie et les faits du texte original.
  Tu n'ajoutes AUCUNE information factuelle nouvelle.
  Même si tu as des connaissances extérieures, tu ne dois pas les utiliser pour enrichir le texte.
- Lorsque la durée ou l’ancienneté est indiquée explicitement (par exemple « depuis plus de deux siècles »), tu ne dois pas la reformuler avec des termes qui changent l’ordre de grandeur. Par exemple, tu ne dois pas écrire « tradition millénaire », « depuis des millénaires » ou « depuis des siècles et des siècles » si ces expressions ne figurent pas dans le texte original.
- Tu ne dois pas extrapoler ou commenter la situation d'autres acteurs (familles, maisons, concurrents) au-delà de ce qui est explicitement écrit. Par exemple, tu ne dois pas écrire que "certaines ont prospéré" ou que "d'autres ont disparu" si ces éléments ne figurent pas dans le texte original.
- Tu ne modifies pas le rôle ni l'importance des acteurs (personnes, maisons, marques) :
  tu ne présentes pas le sujet comme ayant “façonné”, “révolutionné” ou “transformé” un secteur si ce n'est pas explicitement mentionné.
- Tu ne supprimes aucune information importante : aucune phrase ou idée significative du texte original ne doit disparaître.
- Tu peux uniquement ajouter des liaisons neutres pour la fluidité (“au fil des décennies”, “peu à peu”, “aujourd'hui”, etc.), sans créer de faits nouveaux.
- Longueur : texte de longueur comparable au texte d'origine (environ ± 20 %).
- Ton : équilibre entre institutionnel et narratif, avec un lyrisme mesuré.
  Tu t'inspires en priorité du ton du texte original.
- Cohérence des temps verbaux : si le texte raconte une histoire passée, tu utilises des temps du passé (passé composé, imparfait, etc.), pas le futur, sauf si le futur est déjà présent dans le texte original.

Structure :
- Le titre de la section (par exemple « Du siècle des Lumières à nos jours ») est fourni comme contexte. TU NE DOIS PAS LE RÉÉCRIRE.
- Tu ne dois pas ajouter ce titre au début de ta réponse, ni le dupliquer, ni le transformer.
- Tu produis un texte continu composé de paragraphes.
- Tu ne crées PAS de sections ou intitulés supplémentaires comme « Introduction », « Conclusion », etc., sauf si ces mots sont déjà présents dans le texte original.
- Tu ne crées PAS de listes à puces si le texte d'origine n'en contient pas.
- Si des listes existent, tu peux les lisser mais SANS ajouter de nouveaux items.

Mise en forme :
- Sortie en texte brut uniquement (pas de Markdown, pas de HTML, pas de gras/italique).
- Pas de commentaires méta du type « Voici le texte réécrit », « Dans cet article », etc.
"""


def build_geo_prompt(
    original_text: str,
    target_query: str,
    rewrite_mode: str = "ameliorer",
) -> str:
    """
    Construit un prompt complet pour le backend LLM (Gemini ou Ollama),
    en injectant :
    - les instructions GEO,
    - un exemple de style,
    - le texte source à reformuler,
    - le mode de réécriture (minimal, améliorer, créatif mesuré).
    """
    original_text = (original_text or "").strip()
    target_query = (target_query or "").strip()

    if not original_text:
        raise ValueError("Le texte original est vide.")
    if not target_query:
        raise ValueError("La requête cible est vide.")

    # Instructions plus directives et distinctes
    rewrite_mode = (rewrite_mode or "ameliorer").lower()
    if rewrite_mode == "minimal":
        mode_instructions = """
MODE : RÉÉCRITURE MINIMALE
Objectif : Corriger, nettoyer et lisser très légèrement, sans transformer le texte.

Règles spécifiques :
1. Corrige uniquement l'orthographe, la grammaire, la typographie et la ponctuation.
2. Tu peux effectuer de très légers ajustements pour la clarté (ordre des mots, répétitions), mais tu dois :
   - conserver le même nombre d'idées,
   - conserver toutes les phrases importantes,
   - ne pas changer le ton de manière notable.
3. Tu ne fusionnes pas et tu ne supprimes pas de phrases complètes.
4. Tu ne modifies pas la structure du texte au-delà de micro-ajustements.
5. INTERDIT :
   - d'ajouter des informations nouvelles,
   - de supprimer des informations ou des phrases significatives,
   - de réorganiser profondément le texte.
"""
    elif rewrite_mode == "creatif":
        mode_instructions = """
MODE : PROPOSITION CRÉATIVE
Objectif : Proposer une version plus engageante du texte, tout en restant 100 % factuelle.

Règles spécifiques :
1. Utilise un vocabulaire plus riche et des tournures plus travaillées, mais reste dans un ton institutionnel et sobre (pas de roman).
2. Tu peux varier le rythme des phrases (alternance de phrases courtes et longues) pour rendre la lecture plus agréable.
3. Tu conserves toutes les informations importantes du texte original : aucune idée ni aucun fait ne doit disparaître.
4. Tu ne modifies pas la portée des affirmations : tu ne présentes pas la Maison ou les personnes comme ayant un rôle plus grand que celui qui est décrit (pas de « a façonné les grandes familles », etc., si ce n'est pas écrit).
5. Tu ne dois pas non plus commenter l'évolution ou la réussite d'autres maisons ou acteurs (survie, prospérité, déclin, disparition, etc.) si cela n'est pas explicitement mentionné dans le texte original.
6. INTERDIT :
   - d'inventer des faits, des événements, des acquisitions, des extensions de domaine, des personnes ou des dates qui ne sont pas dans le texte,
   - d'ajouter des listes ou des sous-titres qui n'existent pas dans le texte original,
   - d'utiliser un ton romanesque ou exagéré (« légende », « empire », « mission sacrée », etc.), sauf si ces termes sont déjà présents dans le texte original.
7. Lorsque le texte mentionne une durée précise ou approximative (par exemple « depuis plus de deux siècles »), tu ne dois pas la amplifier en parlant de « tradition millénaire », « depuis des millénaires » ou d’une ancienneté plus grande. Tu restes strictement dans le même ordre de grandeur que le texte d’origine.
8. Tu respectes toutes les règles globales définies dans GEO_SYSTEM_INSTRUCTIONS.
"""
    else:  # "ameliorer" par défaut
        mode_instructions = """
MODE : AMÉLIORER LA TOURNURE (STANDARD)
Objectif : Rendre le texte plus fluide, professionnel et agréable à lire, sans modifier le sens.

Règles spécifiques :
1. Reformule les phrases lourdes ou maladroites pour améliorer la clarté.
2. Tu peux ajouter des connecteurs logiques naturels (« ainsi », « au fil des ans », « par ailleurs ») pour fluidifier le texte.
3. Tu peux regrouper ou découper des phrases, à condition de conserver toutes les informations du texte original.
4. Tu respectes l'ordre logique des idées et la chronologie décrite dans le texte.
5. Tu ne rajoutes pas de commentaires généraux sur la situation des autres maisons, familles ou acteurs (par exemple « certains ont prospéré », « d'autres ont disparu ») si ces éléments ne sont pas clairement indiqués dans le texte d'origine.
6. Tu ne modifies pas le rôle ni l'importance des acteurs par rapport à ce qui est écrit.
7. INTERDIT :
   - d'ajouter de nouveaux faits ou détails qui ne sont pas présents dans le texte d'origine,
   - de supprimer des phrases ou des idées importantes,
   - de transformer le texte en discours trop lyrique ou romanesque.
"""

    # ... (code existant pour les exemples few-shot) ...

    # Exemple concret (few-shot) pour ancrer le style attendu
    exemple_original = (
        "Au cœur de la Bourgogne, un jeune homme, se lance dans le vin. "
        "Jean-Claude Boisset a 18 ans, la fougue et la passion de la jeunesse et tout l'avenir devant lui… "
        "Nous sommes à deux pas de Gevrey-Chambertin où il a grandi. C'est l'été 1961. "
        "A partir de 1970 les vins s'exportent en Europe. Jean-Claude Boisset s'installe à Vougeot, "
        "son village de prédilection puis à Nuits-Saint-Georges. "
        "Dès les années 1980 est initiée la croissance externe en Bourgogne ainsi que la diversification "
        "de l'activité avec l'introduction des spiritueux en 1988 et des vins effervescents à la veille des années 1990. "
        "Les années qui suivent marquent la volonté de s'implanter au cœur même des terroirs : le Beaujolais, le Languedoc, le Rhône. "
        "Les années 2000, nouveau millénaire, nouveau continent : la Californie. "
        "L'histoire continue avec l'acquisition du Domaine Maire & Fils, le plus grand domaine viticole du Jura, "
        "les Maisons Meffre, Alex Gambal, Chais du Sud et le Grand Courtâge."
    )

    exemple_reecrit = (
        "Notre héritage prend racine au cœur de la Bourgogne. En 1961, à deux pas de Gevrey-Chambertin où il a grandi, "
        "Jean-Claude Boisset, alors âgé de 18 ans, se lance dans le vin, porté par la fougue et la passion de la jeunesse.\n\n"
        "À partir de 1970, ses vins commencent à s'exporter en Europe. Jean-Claude Boisset s'installe d'abord à Vougeot, "
        "son village de prédilection, puis à Nuits-Saint-Georges, ancrant son histoire au plus près des grands terroirs bourguignons.\n\n"
        "Dans les années 1980, une dynamique de croissance externe s'engage en Bourgogne, accompagnée d'une diversification de l'activité "
        "avec l'introduction des spiritueux en 1988, puis des vins effervescents à la veille des années 1990.\n\n"
        "Les années qui suivent marquent la volonté de s'implanter au cœur d'autres vignobles français – Beaujolais, Languedoc, Rhône – "
        "avant l'ouverture vers un nouveau continent dans les années 2000 avec la Californie. "
        "L'histoire continue avec l'acquisition du Domaine Maire & Fils, le plus grand domaine viticole du Jura, "
        "ainsi que des Maisons Meffre, Alex Gambal, Chais du Sud et Le Grand Courtâge."
    )

    prompt = f"""{GEO_SYSTEM_INSTRUCTIONS}

Requête cible (titre de section ou intention principale) :
"{target_query}"

Exemple de réécriture attendue (pour contexte de style) :

Texte original (exemple) :
---
{exemple_original}
---

Texte réécrit (exemple de version GEO optimisée) :
---
{exemple_reecrit}
---

Cet exemple illustre le niveau de réécriture attendu pour le mode « Améliorer la tournure ». Si le mode demandé est « Réécriture minimale » ou « Proposition créative », adapte ton degré de modification en suivant STRICTEMENT les règles du mode correspondant.

Maintenant, applique la logique au texte ci-dessous.

Texte original à reformuler :
---
{original_text}
---


{mode_instructions}

Consigne finale (très importante) :
- Tu ne dois pas réécrire le titre "{target_query}" : il sert uniquement de contexte.
- Tu réécris UNIQUEMENT le texte ci-dessous.
- Applique STRICTEMENT les consignes du mode choisi.
- Pas de commentaires, pas de balises.
"""

    return prompt


def geo_rewrite_content(
    original_text: str,
    target_query: str,
    model_name: Optional[str] = None,
    rewrite_mode: str = "ameliorer",
    backend: str = "ollama",
    user_api_key: Optional[str] = None,
) -> str:
    """
    Point d'entrée principal pour la reformulation GEO.

    Args:
        original_text: Le texte source à reformuler.
        target_query: Titre de section ou requête cible (contexte).
        model_name: Nom du modèle (optionnel).
        rewrite_mode: Mode de réécriture :
            - "minimal": Correction et lissage léger.
            - "ameliorer": Fluidité et ton professionnel (défaut).
            - "creatif": Style plus riche mais 100% factuel.
        backend: "ollama" ou "gemini".
        user_api_key: Clé API optionnelle pour Gemini.
    """
    if original_text and len(original_text) > 10000:
        raise ValueError(
            "Le texte est trop long pour une reformulation en une seule fois (max 10k caractères). "
            "Merci de le raccourcir ou de le traiter en plusieurs parties."
        )

    rewrite_mode = (rewrite_mode or "ameliorer").lower()
    prompt = build_geo_prompt(original_text, target_query, rewrite_mode=rewrite_mode)
    backend = (backend or "ollama").lower()

    # On garde une température interne, mais elle n'est plus exposée à l'utilisateur
    if rewrite_mode == "minimal":
        temperature = 0.0  # Très déterministe pour le minimal
    elif rewrite_mode == "creatif":
        temperature = 0.5  # Plus élevé pour permettre la variation
    else:  # "ameliorer"
        temperature = 0.3  # Standard

    if backend == "ollama":
        model = model_name or OLLAMA_MODEL_NAME
        return call_ollama_chat(
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

    if backend == "gemini":
        configure_gemini(api_key=user_api_key)
        model_id = model_name or DEFAULT_GEMINI_MODEL
        model = get_gemini_model(model_id)
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=float(temperature),
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Erreur lors de l'appel à Gemini : {exc}") from exc

        text = getattr(response, "text", None)

        if not text:
            # Fallback : tenter de reconstruire le texte à partir des candidats / parts
            fallback_chunks = []

            candidates = getattr(response, "candidates", None)
            if candidates:
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if parts:
                        for part in parts:
                            part_text = getattr(part, "text", None)
                            if part_text:
                                fallback_chunks.append(part_text)

            text = " ".join(fallback_chunks).strip() if fallback_chunks else ""

        if not text:
            raise RuntimeError("Réponse vide ou bloquée par Gemini (filtres de sécurité ?).")

        return text.strip()

    raise ValueError(f"Backend LLM inconnu : {backend}")


def test_gemini_connection(
    model_name: Optional[str] = None,
    user_api_key: Optional[str] = None,
) -> str:
    """
    Teste la connexion à Gemini :
    - configuration avec la clé fournie ou .env
    - génération d'une petite réponse
    """
    try:
        configure_gemini(api_key=user_api_key)
        model_id = model_name or DEFAULT_GEMINI_MODEL
        model = get_gemini_model(model_id)
        resp = model.generate_content("Réponds simplement : 'OK Gemini GEO'.")
        text = getattr(resp, "text", "") or ""
        if "OK Gemini GEO" in text:
            return "Connexion à Gemini OK ✅"
        return "Gemini répond, mais le message de test ne correspond pas exactement (OK Gemini GEO)."
    except Exception as exc:
        return f"Connexion à Gemini KO ❌ : {exc}"


# =========================
# GEO MONITORING (DuckDuckGo)
# =========================

@dataclasses.dataclass
class MonitoringResult:
    query: str
    rank: int
    title: str
    url: str
    snippet: str
    brand_present: bool


DDG_SEARCH_URL = "https://duckduckgo.com/html/"


def _fetch_duckduckgo_results(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Scraping léger de DuckDuckGo (interface HTML).
    Important : usage limité, respectueux, pour un monitoring simple / POC.

    Retourne une liste de dicts bruts avec titre, url, snippet.
    """
    params = {"q": query, "kl": "fr-fr"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            DDG_SEARCH_URL, params=params, headers=headers, timeout=15
        )
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    # La structure HTML peut changer : on reste volontairement simple/robuste.
    for res in soup.select("div.result"):
        link = res.select_one("a.result__a")
        snippet_el = res.select_one("a.result__snippet") or res.select_one(
            "div.result__snippet"
        )

        if not link:
            continue

        title = link.get_text(" ", strip=True)
        url = link.get("href", "")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

        if len(results) >= max_results:
            break

    return results


def monitor_keywords(
    queries: List[str],
    brand_or_domain: str,
    max_results: int = 10,
) -> pd.DataFrame:
    """
    Monitoring simple :
    - queries : liste de requêtes à tester
    - brand_or_domain : nom de marque ou domaine (ex : "boisset.com")
    - max_results : nombre max de résultats analysés par requête

    Retourne un DataFrame avec colonnes :
    - query, rank, title, url, snippet, brand_present
    """
    brand = (brand_or_domain or "").strip().lower()
    rows: List[MonitoringResult] = []

    for query in queries:
        q = (query or "").strip()
        if not q:
            continue

        raw_results = _fetch_duckduckgo_results(q, max_results=max_results)

        for idx, r in enumerate(raw_results, start=1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")

            text_concat = " ".join([title, url, snippet]).lower()
            brand_present = bool(brand and brand in text_concat)

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

    df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    return df

