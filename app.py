import json

import streamlit as st

from config import (
    GEO_ENV,
    IS_PROD,
    DEFAULT_GEMINI_MODEL,
    OLLAMA_MODEL_NAME,
    DEFAULT_BACKEND,
)
from geo_utils import (
    geo_rewrite_content,
    monitor_keywords,
    test_gemini_connection,
    test_ollama_connection,
)


# -----------------------------------------------------------------------------
# CONFIG GLOBALE STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GEO Architect - Assistant MVP GEO",
    layout="wide",
)


# -----------------------------------------------------------------------------
# BLOCS UTILITAIRES
# -----------------------------------------------------------------------------
def render_backend_diagnostics() -> None:
    """
    Petit bloc de diagnostic pour vérifier rapidement les backends LLM.

    - En environnement PROD : on ne montre rien (interface plus simple).
    - En DEV : permet de tester Ollama et Gemini.
    """
    if IS_PROD:
        return

    with st.expander("🔍 Diagnostics des backends LLM"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Ollama (local)")
            st.code(f"Base URL : {st.session_state.get('OLLAMA_BASE_URL', 'http://localhost:11434')}\n"
                    f"Modèle : {OLLAMA_MODEL_NAME}")
            if st.button("🧪 Tester Ollama", key="test_ollama"):
                msg = test_ollama_connection()
                if "OK" in msg:
                    st.success(msg)
                else:
                    st.error(msg)

        with col2:
            st.markdown("#### Gemini (cloud)")
            st.write(
                "Le test utilisera la clé API définie dans les secrets Streamlit "
                "ou celle saisie ci-dessous."
            )
            user_key = st.text_input(
                "Clé API Gemini (optionnelle pour le test)",
                type="password",
                key="diag_gemini_key",
            )
            model_name = st.text_input(
                "Modèle Gemini à tester",
                value=DEFAULT_GEMINI_MODEL,
                key="diag_gemini_model",
            )
            if st.button("🧪 Tester Gemini", key="test_gemini"):
                msg = test_gemini_connection(
                    model_name=model_name,
                    user_api_key=user_key or None,
                )
                if "OK" in msg:
                    st.success(msg)
                else:
                    st.error(msg)


# -----------------------------------------------------------------------------
# ONGLET GEO REFORMULATION
# -----------------------------------------------------------------------------
def render_geo_reformulation_tab() -> None:
    """
    Interface principale de reformulation GEO.

    Objectifs d'UI :
    - Une bannière d'information claire en haut.
    - Un "carton" d'intro GEO Reformulation pleine largeur.
    - En dessous, grille 2 colonnes :
        - À gauche (2/3) : Contenu à optimiser + Niveau de réécriture.
        - À droite (1/3) : Texte GEO optimisé + bouton "Générer".
      La carte de droite occupe visuellement la hauteur des deux cartes de gauche.
    - Respect automatique du mode sombre / clair de Streamlit (pas de CSS qui force un fond).
    """

    # Bannière d’avertissement sur l’usage de Gemini
    st.info(
        "Cette version de GEO Architect utilise l'API Gemini en mode cloud. "
        "Ne collez pas de données sensibles ou strictement confidentielles.",
        icon="ℹ️",
    )

    # Carton d'intro pleine largeur
    with st.container(border=True):
        left, right = st.columns([0.1, 0.9])
        with left:
            st.markdown("### ✨")
        with right:
            st.markdown("### GEO Reformulation")
            st.write(
                "Colle un contenu existant, renseigne un titre de section (ou une requête cible simple), "
                "puis laisse l'assistant générer une version optimisée GEO : neutre, factuelle, structurée, "
                "prête à être exploitée par des moteurs IA."
            )

    # Sélection du backend :
    # - En PROD : toujours Gemini, on ne montre pas l'option.
    # - En DEV : possibilité de choisir entre Ollama et Gemini.
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
            backend = "ollama" if "Ollama" in backend_choice else "gemini"
            if backend == "ollama":
                st.caption(f"Modèle local : `{OLLAMA_MODEL_NAME}` (nécessite Ollama lancé en local).")
            else:
                st.caption(f"Modèle cloud : `{DEFAULT_GEMINI_MODEL}`.")

    # Initialisation du state pour le résultat
    if "geo_result" not in st.session_state:
        st.session_state["geo_result"] = ""
    if "geo_result_area" not in st.session_state:
        st.session_state["geo_result_area"] = ""
    if "previous_rewrite_mode" not in st.session_state:
        st.session_state["previous_rewrite_mode"] = "ameliorer"

    # Mapping des labels -> valeurs internes
    mode_label_to_value = {
        "Réécriture minimale": "minimal",
        "Améliorer la tournure": "ameliorer",
        "Proposition créative": "creatif",
    }

    # Grille principale : gauche (inputs) / droite (résultat)
    col_main, col_result = st.columns([2, 1])

    # -------------------------
    # COLONNE GAUCHE
    # -------------------------
    with col_main:
        # Carte "Contenu à optimiser"
        with st.container(border=True):
            st.markdown("#### Contenu à optimiser")

            target_query = st.text_input(
                "Titre de section (ou requête cible simple)",
                placeholder='"titre", "meta-titre", "slug", "meta-description", "h1", "h2", "h3", "faq"...',
                key="geo_target_query",
            )

            original_text = st.text_area(
                "Texte original",
                height=260,
                placeholder=(
                    "Coller le texte brut ici, par exemple un contenu créé par un autre service, "
                    "Google Trends, ChatGPT..."
                ),
                key="geo_original_text",
            )

        # Carte "Niveau de réécriture"
        with st.container(border=True):
            st.markdown("#### Niveau de réécriture")

            mode_label = st.selectbox(
                "Choix du niveau",
                list(mode_label_to_value.keys()),
                index=1,  # "Améliorer la tournure" par défaut
                key="geo_rewrite_mode_label",
            )
            rewrite_mode = mode_label_to_value[mode_label]

            st.caption(
                "• **Réécriture minimale** : corrections et ajustements très légers.\n"
                "• **Améliorer la tournure** : texte plus fluide, même contenu.\n"
                "• **Proposition créative** : style plus travaillé, toujours factuel."
            )

            # Reset du résultat si le mode change
            if st.session_state["previous_rewrite_mode"] != rewrite_mode:
                st.session_state["previous_rewrite_mode"] = rewrite_mode
                st.session_state["geo_result"] = ""
                st.session_state["geo_result_area"] = ""

    # S'assurer que la zone de résultat part de la bonne valeur
    st.session_state["geo_result_area"] = st.session_state.get(
        "geo_result",
        st.session_state.get("geo_result_area", ""),
    )

    # -------------------------
    # COLONNE DROITE
    # -------------------------
    generate_button_clicked = False

    with col_result:
        with st.container(border=True):
            header_cols = st.columns([0.8, 0.2])
            with header_cols[0]:
                st.markdown("#### Texte GEO optimisé")
            with header_cols[1]:
                # petit indicateur "prêt" si on a déjà un texte
                if st.session_state.get("geo_result"):
                    st.markdown("✅\n\n*Texte prêt*")
                else:
                    st.markdown("📝\n\n*En attente*")

            result_text = st.text_area(
                "Résultat",
                height=320,
                key="geo_result_area",
            )
            # Synchronisation avec l'état interne
            st.session_state["geo_result"] = result_text

            st.markdown("---")
            generate_button_clicked = st.button(
                "✨ Générer",
                type="primary",
                use_container_width=True,
                key="geo_generate_button",
            )

            if result_text:
                st.success("Texte optimisé généré avec succès !")

    # -------------------------
    # LOGIQUE D'APPEL LLM (APRÈS LE LAYOUT)
    # -------------------------
    if generate_button_clicked:
        if not original_text or not original_text.strip():
            st.warning("Merci de coller un texte à reformuler.")
            return

        if not target_query or not target_query.strip():
            st.warning("Merci de préciser un titre de section ou une requête cible.")
            return

        with st.spinner("Génération de la version GEO en cours..."):
            try:
                rewritten = geo_rewrite_content(
                    original_text=original_text,
                    target_query=target_query,
                    model_name=None,  # Laisse geo_utils choisir le modèle par défaut
                    rewrite_mode=rewrite_mode,
                    backend=backend,
                    user_api_key=None,  # Clé gérée côté serveur (secrets / config)
                )
                st.session_state["geo_result"] = rewritten
                st.session_state["geo_result_area"] = rewritten
                # On force un rerun pour rafraîchir proprement la zone de texte & l'indicateur
                st.rerun()
            except Exception as exc:
                st.error(f"Erreur lors de la reformulation : {exc}")

    # Pied de page informatif
    st.caption(
        f"Modèle IA : `{DEFAULT_GEMINI_MODEL}` (Gemini) · "
        f"Environnement : `{GEO_ENV}` · "
        "Backend IA : Gemini (forcé en production)."
    )


# -----------------------------------------------------------------------------
# ONGLET GEO MONITORING (inchangé dans la logique, juste léger polish UI)
# -----------------------------------------------------------------------------
def render_geo_monitoring_tab() -> None:
    st.header("📊 GEO Monitoring (simple)")

    st.write(
        "Fournis une ou plusieurs requêtes et une marque / un domaine. "
        "L'application interroge DuckDuckGo (scraping léger) et indique si la marque "
        "apparaît dans les premiers résultats."
    )

    st.caption(
        "⚠️ Ce monitoring est volontairement simple et limité. "
        "Il ne doit pas être utilisé pour du scraping massif "
        "(risque de blocage / non-respect des conditions d'utilisation des moteurs)."
    )

    with st.container(border=True):
        with st.form("geo_monitoring_form"):
            queries_text = st.text_area(
                "Liste de requêtes (une par ligne)",
                height=200,
                placeholder="Ex :\nmaison boisset avis\nmaison boisset histoire\nmaison boisset bourgogne",
            )
            brand_or_domain = st.text_input(
                "Marque ou domaine à détecter",
                placeholder="Ex : boisset, boisset.com...",
            )
            max_results = st.slider(
                "Nombre max de résultats analysés par requête",
                min_value=3,
                max_value=20,
                value=10,
            )

            submitted = st.form_submit_button("🔍 Lancer le monitoring")

        if not submitted:
            return

    # Traitement une fois le formulaire soumis
    queries = [
        line.strip()
        for line in (queries_text or "").splitlines()
        if line.strip()
    ]

    if not queries:
        st.warning("Merci de saisir au moins une requête.")
        return
    if not brand_or_domain.strip():
        st.warning("Merci de saisir une marque ou un domaine à détecter.")
        return

    with st.spinner("Interrogation de DuckDuckGo en cours..."):
        df = monitor_keywords(
            queries=queries,
            brand_or_domain=brand_or_domain,
            max_results=max_results,
        )

    if df.empty:
        st.info("Aucun résultat n'a été trouvé ou le scraping a échoué.")
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
    synth["présence_marque"] = synth["brand_present_any"].map(
        lambda v: "✅ présente" if v else "❌ absente"
    )
    st.dataframe(synth[["query", "présence_marque"]], use_container_width=True)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("GEO Architect – Assistant MVP GEO")

    # Diagnostics LLM (seulement en DEV)
    render_backend_diagnostics()

    tab1, tab2 = st.tabs(["🧠 GEO Reformulation", "📊 GEO Monitoring"])

    with tab1:
        render_geo_reformulation_tab()
    with tab2:
        render_geo_monitoring_tab()


if __name__ == "__main__":
    main()
