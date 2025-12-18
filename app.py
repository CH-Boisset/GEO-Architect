import time
import hashlib
import json
from typing import Any, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="GEO Architect", page_icon="🧠", layout="wide")

from config import (
    GEO_ENV,
    IS_PROD,
    DEFAULT_GEMINI_MODEL,
    OLLAMA_MODEL_NAME,
    DEFAULT_BACKEND,
    OLLAMA_BASE_URL,
    ADMIN_UI_TOKEN,
)

from geo_utils import (
    geo_rewrite_content,
    geo_is_text_already_optimized,
    monitor_keywords,
    test_gemini_connection,
    test_ollama_connection,
)

# -----------------------------------------------------------------------------
# UI sizing (évite le scroll initial)
# -----------------------------------------------------------------------------
UI_ORIGINAL_TEXT_HEIGHT = 260
UI_RESULT_TEXT_HEIGHT = 360
UI_MONITOR_QUERIES_HEIGHT = 180

# -----------------------------------------------------------------------------
# Admin UI (masquer toolbar/menu Streamlit pour users, garder pour admin)
# -----------------------------------------------------------------------------

def _get_query_param(name: str) -> Optional[str]:
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        val = qp.get(name)
        if isinstance(val, list):
            return val[0] if val else None
        return val
    except Exception:
        qp = st.experimental_get_query_params()
        val_list = qp.get(name, [])
        return val_list[0] if val_list else None


def _apply_user_css_hide_toolbar() -> None:
    st.markdown(
        """
<style>
#MainMenu {visibility: hidden;}
[data-testid="stToolbarActions"] {display: none !important;}
header [data-testid="stToolbar"] {display: none !important;}
footer {visibility: hidden;}
</style>
        """,
        unsafe_allow_html=True,
    )


def _is_admin_session() -> bool:
    token = (ADMIN_UI_TOKEN or "").strip()
    if not token:
        return False
    admin_qp = _get_query_param("admin") or ""
    return admin_qp == token


# -----------------------------------------------------------------------------
# Session state init (anti double-clic / cache / flags)
# -----------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "rewrite_cache": {},              # sig -> result dict
        "rewrite_inflight": False,
        "pending_action": None,           # "generate"
        "pending_payload": None,          # dict
        "last_request_sig": "",
        "last_request_ts": 0.0,
        "cooldown_until_ts": 0.0,         # 429 cooldown
        "last_result": None,              # dernier dict résultat
        "optimized_gate": None,           # dict si pré-check déclenche
        "force_after_optimized": False,   # si l’utilisateur force
        "pending_set_mode_label": None,   # SAFE: set widget state before widget creation
        "show_post_optimized_modal": False,  # pop-up post-résultat
        "post_optimized_modal_sig": "",      # sig déjà affichée (anti boucle)
        "last_result_sig": "",               # sig du dernier résultat affiché
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _norm_for_sig(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    t = "\n".join(" ".join(line.split()) for line in t.split("\n"))
    return t


def _make_sig(original_text: str, target_query: str, rewrite_mode: str, model_name: str, backend: str) -> str:
    payload = "||".join(
        [
            _norm_for_sig(original_text),
            _norm_for_sig(target_query),
            (rewrite_mode or "").strip().lower(),
            (model_name or "").strip(),
            (backend or "").strip().lower(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def copy_to_clipboard_button(text: str, label: str = "📋 Copier le texte", key: str = "copy_btn") -> None:
    """
    Bouton de copie vers le presse-papiers via un vrai clic DOM (composant HTML).
    - Active clipboard API si possible (HTTPS / user gesture)
    - Fallback execCommand('copy')
    """
    t = (text or "")
    is_disabled = not t.strip()
    disabled_attr = "disabled" if is_disabled else ""
    disabled_js = "true" if is_disabled else "false"

    payload = json.dumps(t)

    # Style simple + état désactivé explicite (sans syntaxe {{...}} invalide)
    btn_style = """
      width: 100%;
      padding: 0.6rem 0.75rem;
      border-radius: 0.5rem;
      border: 1px solid rgba(49, 51, 63, 0.2);
      background: white;
      cursor: pointer;
      font-weight: 600;
    """.strip()

    if is_disabled:
        btn_style += " opacity: 0.5; cursor: not-allowed;"

    components.html(
        f"""
<div style="margin-top:8px;">
  <button id="{key}_btn" {disabled_attr} style="{btn_style}">
    {label}
  </button>
</div>

<script>
(function() {{
  const btn = document.getElementById("{key}_btn");
  if (!btn) return;

  const disabled = {disabled_js};
  if (disabled) {{
    btn.disabled = true;
    return;
  }}

  const textToCopy = {payload};

  async function doCopy() {{
    try {{
      if (navigator.clipboard && window.isSecureContext) {{
        await navigator.clipboard.writeText(textToCopy);
      }} else {{
        const ta = document.createElement("textarea");
        ta.value = textToCopy;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        ta.style.top = "-9999px";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }}
      const old = btn.innerText;
      btn.innerText = "Copié ✅";
      setTimeout(() => {{ btn.innerText = old; }}, 1500);
    }} catch (e) {{
      console.log("Clipboard error:", e);
      const old = btn.innerText;
      btn.innerText = "Copie impossible ❌";
      setTimeout(() => {{ btn.innerText = old; }}, 2000);
    }}
  }}

  btn.addEventListener("click", (e) => {{
    e.preventDefault();
    doCopy();
  }});
}})();
</script>
        """,
        height=90,
    )


# -----------------------------------------------------------------------------
# Diagnostics (DEV)
# -----------------------------------------------------------------------------

def render_backend_diagnostics() -> None:
    if IS_PROD:
        return

    st.divider()
    st.subheader("🧪 Diagnostics (DEV)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Ollama (local)")
        st.caption(f"Base URL : `{OLLAMA_BASE_URL}` · Modèle : `{OLLAMA_MODEL_NAME}`")
        if st.button("Tester Ollama", key="diag_test_ollama"):
            ok, msg = test_ollama_connection()
            st.success(msg) if ok else st.error(msg)

    with col2:
        st.markdown("#### Gemini (cloud)")
        st.write(
            "Le test utilisera la clé API définie dans les variables d'environnement "
            "ou dans les secrets Streamlit."
        )
        user_key = st.text_input("Clé API Gemini (optionnelle pour le test)", type="password", key="diag_gemini_key")
        if st.button("Tester Gemini", key="diag_test_gemini"):
            ok, msg = test_gemini_connection(api_key=user_key or None)
            st.success(msg) if ok else st.error(msg)


# -----------------------------------------------------------------------------
# GEO Reformulation tab
# -----------------------------------------------------------------------------

def render_geo_reformulation_tab() -> None:
    st.info(
        "Cette version de GEO Architect utilise l'API Gemini en mode cloud. "
        "Ne collez pas de données sensibles ou strictement confidentielles.",
        icon="⚠️",
    )

    if IS_PROD:
        backend = "gemini"
    else:
        with st.container(border=True):
            st.markdown("#### Backend IA (DEV uniquement)")
            backend_choice = st.radio(
                "Moteur IA",
                ["Ollama (Local)", "Gemini (Cloud)"],
                index=0 if DEFAULT_BACKEND == "ollama" else 1,
                horizontal=True,
                key="backend_radio",
            )
        backend = "ollama" if backend_choice.startswith("Ollama") else "gemini"

    col_main, col_result = st.columns([2, 1])

    with col_main:
        with st.container(border=True):
            st.markdown("#### Contenu à optimiser")

            target_query = st.text_input(
                "Titre de section (ou intention / requête cible)",
                placeholder="Ex : histoire de Maison Boisset, qui est Jean-Charles Boisset, etc.",
                key="geo_target_query",
            )

            original_text = st.text_area(
                "Texte original",
                height=UI_ORIGINAL_TEXT_HEIGHT,
                placeholder="Collez ici le texte à reformuler (texte brut).",
                key="geo_original_text",
            )

    # SAFE: appliquer une éventuelle demande de mode AVANT la création du widget selectbox
    if st.session_state.get("pending_set_mode_label"):
        st.session_state["geo_rewrite_mode_label"] = st.session_state["pending_set_mode_label"]
        st.session_state["pending_set_mode_label"] = None

    if st.session_state["force_after_optimized"]:
        mode_label_to_value = {
            "Réécriture minimale (Conserver au maximum le texte d'origine)": "minimal",
            "Améliorer la tournure (Modification du texte d'origine)": "ameliorer",
            "Proposition créative (Proposition très différente du texte d'origine)": "creatif",
        }
        mode_labels = list(mode_label_to_value.keys())
        default_label = "Réécriture minimale (Conserver au maximum le texte d'origine)"
    else:
        mode_label_to_value = {
            "Réécriture minimale": "minimal",
            "Améliorer la tournure": "ameliorer",
            "Proposition créative": "creatif",
        }
        mode_labels = list(mode_label_to_value.keys())
        default_label = "Améliorer la tournure"

    default_index = mode_labels.index(default_label)

    with col_main:
        with st.container(border=True):
            st.markdown("#### Niveau de réécriture")

            # IMPORTANT : ne pas fournir index=... si le widget est déjà piloté via session_state (clé existante)
            if "geo_rewrite_mode_label" in st.session_state and st.session_state["geo_rewrite_mode_label"] in mode_labels:
                mode_label = st.selectbox(
                    "Choix du niveau",
                    mode_labels,
                    key="geo_rewrite_mode_label",
                )
            else:
                mode_label = st.selectbox(
                    "Choix du niveau",
                    mode_labels,
                    index=default_index,
                    key="geo_rewrite_mode_label",
                )

            rewrite_mode = mode_label_to_value[mode_label]
            if st.session_state["force_after_optimized"]:
                st.caption("Vous avez choisi de reformuler malgré l'alerte \"Texte déjà optimisé\".")

    now = time.time()
    cooldown_remaining = max(0, int(st.session_state["cooldown_until_ts"] - now))

    cooldown_placeholder = st.empty()
    if cooldown_remaining > 0:
        cooldown_placeholder.warning(
            f"Quota / limitation détectée. Réessayez dans {cooldown_remaining} seconde(s).",
            icon="⏳",
        )

        # Minuteur live sans dépendance externe :
        # on force un rerun après 1 seconde tant que le cooldown est actif.
        time.sleep(1)
        st.rerun()
    else:
        # Nettoie si on est passé à zéro
        if st.session_state.get("cooldown_until_ts", 0) != 0:
            st.session_state["cooldown_until_ts"] = 0
        cooldown_placeholder.empty()

    gate = st.session_state.get("optimized_gate")
    gate_matches = False
    if isinstance(gate, dict):
        current_sig = _make_sig(original_text, target_query, rewrite_mode, DEFAULT_GEMINI_MODEL, backend)
        gate_matches = bool(gate.get("sig") and gate.get("sig") == current_sig)

    # ---------------------------------------------------------------------
    # POP-UP "Texte déjà optimisé" (UX)
    # ---------------------------------------------------------------------
    if gate_matches:
        @st.dialog("🧠 Texte déjà optimisé")
        def _optimized_modal():
            st.markdown("### Texte déjà optimisé")
            st.write(
                "Le texte que vous venez de coller est déjà optimisé pour le référencement dans les IA."
            )
            st.write("Voulez-vous quand même obtenir une reformulation ?")

            st.info(
                "Si vous choisissez de reformuler, vous pourrez sélectionner l’un des 3 niveaux : "
                "réécriture minimale, amélioration de la tournure, ou proposition créative.",
                icon="ℹ️",
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("Ne pas reformuler (recommandé)", type="primary", use_container_width=True, key="modal_skip_rewrite"):
                    st.session_state["last_result"] = {
                        "text": original_text,
                        "already_optimized": True,
                        "similarity": 1.0,
                        "repaired": False,
                        "violations": [],
                        "cooldown_seconds": None,
                        "error": None,
                    }
                    st.session_state["optimized_gate"] = None
                    st.session_state["force_after_optimized"] = False

                    # Déclenche aussi la pop-up post-résultat (plus visible) après affichage du texte
                    st.session_state["last_result_sig"] = current_sig
                    if st.session_state.get("post_optimized_modal_sig") != current_sig:
                        st.session_state["show_post_optimized_modal"] = True
                        st.session_state["post_optimized_modal_sig"] = current_sig

                    st.rerun()

            with c2:
                if st.button("Reformuler quand même", use_container_width=True, key="modal_force_rewrite"):
                    # Active le mode "post-alerte" : libellés détaillés côté UI
                    st.session_state["force_after_optimized"] = True
                    # SAFE: ne pas modifier un widget après instanciation -> on utilise pending_set_mode_label
                    st.session_state["pending_set_mode_label"] = "Réécriture minimale (Conserver au maximum le texte d'origine)"
                    st.session_state["optimized_gate"] = None
                    st.rerun()

            with c3:
                if st.button("Continuer à éditer", use_container_width=True, key="modal_continue_edit"):
                    # On ferme la pop-up en retirant le gate
                    st.session_state["optimized_gate"] = None
                    st.session_state["force_after_optimized"] = False
                    st.rerun()

        _optimized_modal()

    # ---------------------------------------------------------------------
    # POP-UP post-résultat : "Texte déjà optimisé" (quand already_optimized=True)
    # ---------------------------------------------------------------------
    last_res = st.session_state.get("last_result") or {}
    if (
        st.session_state.get("show_post_optimized_modal")
        and isinstance(last_res, dict)
        and last_res.get("already_optimized")
        and not gate_matches  # évite les modals empilées
    ):

        @st.dialog("🧠 Texte déjà optimisé")
        def _post_optimized_modal():
            st.markdown("### Le texte déjà optimisé")
            st.write("pour le référencement dans les IA.")
            st.write("Souhaitez-vous simplement le récupérer, ou relancer une reformulation ?")

            c1, c2 = st.columns(2)

            with c1:
                if st.button("Voir mon texte", type="primary", use_container_width=True, key="postopt_view_text"):
                    st.session_state["show_post_optimized_modal"] = False
                    st.rerun()

            with c2:
                if st.button("Relancer une reformulation", use_container_width=True, key="postopt_change_level"):
                    # Ferme la pop-up
                    st.session_state["show_post_optimized_modal"] = False

                    # Active libellés détaillés et sélectionne "Améliorer..." via pending_set_mode_label
                    st.session_state["force_after_optimized"] = True
                    st.session_state["pending_set_mode_label"] = "Améliorer la tournure (Modification du texte d'origine)"

                    # Relance auto en mode "ameliorer"
                    desired_mode = "ameliorer"
                    sig2 = _make_sig(original_text, target_query, desired_mode, DEFAULT_GEMINI_MODEL, backend)

                    # Si cooldown actif, on laisse l’utilisateur attendre le minuteur (pas de crash)
                    now2 = time.time()
                    cooldown_remaining2 = max(0, int(st.session_state.get("cooldown_until_ts", 0) - now2))
                    if cooldown_remaining2 > 0:
                        st.warning(f"Quota / limitation détectée. Réessayez dans {cooldown_remaining2} seconde(s).", icon="⏳")
                        return

                    # Cache : si déjà présent, on applique sans appel IA
                    cache = st.session_state.get("rewrite_cache", {})
                    if isinstance(cache, dict) and sig2 in cache:
                        st.session_state["last_result"] = cache[sig2]
                        st.session_state["last_result_sig"] = sig2
                        st.rerun()

                    # Schedule génération via pipeline existant
                    st.session_state["pending_action"] = "generate"
                    st.session_state["pending_payload"] = {
                        "sig": sig2,
                        "original_text": original_text,
                        "target_query": target_query,
                        "rewrite_mode": desired_mode,
                        "backend": backend,
                        "model_name": DEFAULT_GEMINI_MODEL,
                    }
                    st.session_state["rewrite_inflight"] = True
                    st.rerun()

        _post_optimized_modal()

    with col_result:
        with st.container(border=True):
            st.markdown("#### Texte GEO optimisé")

            disable_generate = st.session_state["rewrite_inflight"] or cooldown_remaining > 0 or gate_matches
            btn_label = "🧠 Générer" if not st.session_state["force_after_optimized"] else "🧠 Lancer la reformulation"

            if st.button(btn_label, type="primary", disabled=disable_generate, key="btn_generate"):
                sig = _make_sig(original_text, target_query, rewrite_mode, DEFAULT_GEMINI_MODEL, backend)

                if (time.time() - float(st.session_state["last_request_ts"])) < 2.0 and sig == st.session_state["last_request_sig"]:
                    st.info("Requête identique déjà en cours / trop rapprochée. Patientez une seconde.", icon="🛑")
                else:
                    st.session_state["last_request_sig"] = sig
                    st.session_state["last_request_ts"] = time.time()

                    if (not st.session_state["force_after_optimized"]) and target_query.strip() and original_text.strip():
                        if geo_is_text_already_optimized(original_text=original_text, target_query=target_query):
                            st.session_state["optimized_gate"] = {"sig": sig}
                            st.rerun()

                    cache: Dict[str, Any] = st.session_state["rewrite_cache"]
                    if sig in cache:
                        st.session_state["last_result"] = cache[sig]
                        st.rerun()

                    st.session_state["pending_action"] = "generate"
                    st.session_state["pending_payload"] = {
                        "sig": sig,
                        "original_text": original_text,
                        "target_query": target_query,
                        "rewrite_mode": rewrite_mode,
                        "backend": backend,
                        "model_name": DEFAULT_GEMINI_MODEL,
                    }
                    st.session_state["rewrite_inflight"] = True
                    st.rerun()

            if st.session_state.get("pending_action") == "generate" and isinstance(st.session_state.get("pending_payload"), dict):
                payload = st.session_state["pending_payload"]
                st.session_state["pending_action"] = None
                st.session_state["pending_payload"] = None

                with st.spinner("Génération de la version GEO en cours..."):
                    try:
                        res = geo_rewrite_content(
                            original_text=payload["original_text"],
                            target_query=payload["target_query"],
                            model_name=payload.get("model_name"),
                            rewrite_mode=payload["rewrite_mode"],
                            backend=payload["backend"],
                            user_api_key=None,
                        )

                        cd = res.get("cooldown_seconds")
                        if cd:
                            st.session_state["cooldown_until_ts"] = time.time() + int(cd)

                        st.session_state["last_result"] = res
                        st.session_state["rewrite_cache"][payload["sig"]] = res

                        # Mémorise la signature du dernier résultat
                        st.session_state["last_result_sig"] = payload["sig"]

                        # Pop-up post-résultat: si already_optimized, on l’affiche une seule fois par sig
                        if res.get("already_optimized"):
                            if st.session_state.get("post_optimized_modal_sig") != payload["sig"]:
                                st.session_state["show_post_optimized_modal"] = True
                                st.session_state["post_optimized_modal_sig"] = payload["sig"]

                    except Exception as exc:
                        st.session_state["last_result"] = {
                            "text": payload.get("original_text", ""),
                            "already_optimized": False,
                            "similarity": 0.0,
                            "repaired": False,
                            "violations": [],
                            "cooldown_seconds": None,
                            "error": f"Erreur lors de la reformulation : {exc}",
                        }
                    finally:
                        st.session_state["rewrite_inflight"] = False
                        st.session_state["force_after_optimized"] = False

                st.rerun()

            last = st.session_state.get("last_result") or {}
            if isinstance(last, dict):
                if last.get("repaired"):
                    st.caption("🛠️ **Sortie réparée**")

            result_text = ""
            if isinstance(last, dict):
                result_text = (last.get("text") or "")

            st.text_area("Résultat", height=UI_RESULT_TEXT_HEIGHT, value=result_text, disabled=True)
            copy_to_clipboard_button(result_text, key="copy_geo_optimized_text")

            if isinstance(last, dict) and last.get("error"):
                st.error(last["error"])
            if isinstance(last, dict) and last.get("violations"):
                with st.expander("Voir les contrôles de conformité (violations)"):
                    st.write(last.get("violations"))

    st.caption(f"Modèle IA : `{DEFAULT_GEMINI_MODEL}` · Environnement : `{GEO_ENV}` · Backend IA : Gemini (forcé en production).")


# -----------------------------------------------------------------------------
# GEO Monitoring tab
# -----------------------------------------------------------------------------

def render_geo_monitoring_tab() -> None:
    st.header("📊 GEO Monitoring (simple)")
    st.write(
        "Monitoring simple des résultats (DuckDuckGo HTML). "
        "Limites : blocages, CAPTCHA, variations HTML, etc."
    )

    with st.form("monitoring_form"):
        queries_text = st.text_area(
            "Requêtes (1 par ligne)",
            height=UI_MONITOR_QUERIES_HEIGHT,
            placeholder="Ex:\nmaison boisset avis\nmaison boisset histoire\nmaison boisset bourgogne",
            key="monitor_queries",
        )
        brand_or_domain = st.text_input(
            "Marque ou domaine à chercher (dans title / URL / snippet)",
            placeholder="Ex : boisset.com ou 'Maison Boisset'",
            key="monitor_brand",
        )
        max_results = st.slider("Nombre max de résultats analysés par requête", 3, 20, 10)
        submitted = st.form_submit_button("🔍 Lancer le monitoring")

    if not submitted:
        return

    queries = [line.strip() for line in (queries_text or "").splitlines() if line.strip()]
    if not queries:
        st.warning("Ajoutez au moins une requête.", icon="⚠️")
        return
    if not (brand_or_domain or "").strip():
        st.warning("Ajoutez une marque ou un domaine à chercher.", icon="⚠️")
        return

    with st.spinner("Analyse des résultats DuckDuckGo..."):
        try:
            df = monitor_keywords(queries=queries, brand_or_domain=brand_or_domain, max_results=max_results)
        except Exception as exc:
            st.error(f"Erreur monitoring : {exc}")
            return

    if df is None or df.empty:
        st.info("Aucun résultat (ou blocage / HTML inattendu).")
        return

    st.subheader("Résultats détaillés")
    st.dataframe(df, use_container_width=True)

    st.subheader("Synthèse par requête")
    synth = (
        df.groupby("query")["brand_present"]
        .any()
        .reset_index()
        .rename(columns={"brand_present": "brand_present_any"})
    )
    synth["présence_marque"] = synth["brand_present_any"].map(lambda v: "✅ présente" if v else "❌ absente")
    st.dataframe(synth[["query", "présence_marque"]], use_container_width=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    _init_state()

    if not _is_admin_session():
        _apply_user_css_hide_toolbar()

    # --- UI: réduire le bandeau vide en haut (padding container) ---
    st.markdown(
        """
<style>
/* 1) Header Streamlit: invisible + pas de hauteur (pour remonter le contenu) */
header[data-testid="stHeader"]{
  opacity: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  pointer-events: none !important;
}

/* 2) Container global: pas de padding top */
[data-testid="stAppViewContainer"]{
  padding-top: 0 !important;
}

/* 3) Main block container: padding top 0 et padding left/right 40px */
[data-testid="stMainBlockContainer"]{
  padding-top: 0 !important;
  padding-left: 40px !important;
  padding-right: 40px !important;
}

/* Optionnel: éviter une marge résiduelle sous le header si jamais */
[data-testid="stAppViewContainer"] > .main{
  padding-top: 0 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div style="margin-top:0.2rem; margin-bottom:0.25rem;">
  <h1 style="margin:0; padding:0; line-height:1.05;">GEO Architect</h1>
</div>
<div style="margin-top:0; margin-bottom:0.75rem; color: rgba(49,51,63,0.7); font-size:0.95rem;">
  MVP · Reformulation GEO + Monitoring simple
</div>
        """,
        unsafe_allow_html=True,
    )

    render_backend_diagnostics()

    tab1, tab2 = st.tabs(["🧠 GEO Reformulation", "📊 GEO Monitoring"])
    with tab1:
        render_geo_reformulation_tab()
    with tab2:
        render_geo_monitoring_tab()


if __name__ == "__main__":
    main()
