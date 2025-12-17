import streamlit as st

st.set_page_config(page_title="GEO Architect", page_icon="🧠", layout="wide")

from config import (
    GEO_ENV,
    IS_PROD,
    DEFAULT_GEMINI_MODEL,
    OLLAMA_MODEL_NAME,
    DEFAULT_BACKEND,
    OLLAMA_BASE_URL,
)
from geo_utils import (
    geo_rewrite_content,
    geo_is_text_already_optimized,
    monitor_keywords,
    test_gemini_connection,
    test_ollama_connection,
)


# -----------------------------------------------------------------------------
# CONFIG GLOBALE STREAMLIT
# -----------------------------------------------------------------------------
# Rien ici : st.set_page_config est déjà appelé tout en haut (exigence Streamlit).


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

    st.divider()
    st.subheader("🧪 Diagnostics (DEV)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Ollama (local)")
        st.caption(f"Base URL : `{OLLAMA_BASE_URL}` · Modèle : `{OLLAMA_MODEL_NAME}`")
        if st.button("Tester Ollama", key="diag_test_ollama"):
            ok, msg = test_ollama_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with col2:
        st.markdown("#### Gemini (cloud)")
        st.write(
            "Le test utilisera la clé API définie dans les variables d'environnement "
            "ou dans les secrets Streamlit."
        )
        user_key = st.text_input(
            "Clé API Gemini (optionnelle pour le test)",
            type="password",
            key="diag_gemini_key",
        )
        if st.button("Tester Gemini", key="diag_test_gemini"):
            ok, msg = test_gemini_connection(api_key=user_key or None)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


# -----------------------------------------------------------------------------
# ONGLET GEO REFORMULATION
# -----------------------------------------------------------------------------
def render_geo_reformulation_tab() -> None:
    """
    Interface principale de reformulation GEO.
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
                st.caption(
                    f"Modèle local : `{OLLAMA_MODEL_NAME}` "
                    "(nécessite Ollama lancé en local)."
                )
            else:
                st.caption(f"Modèle cloud : `{DEFAULT_GEMINI_MODEL}`.")

    # Initialisation du state pour le résultat
    if "geo_result" not in st.session_state:
        st.session_state["geo_result"] = ""
    if "geo_result_area" not in st.session_state:
        st.session_state["geo_result_area"] = ""
    if "previous_rewrite_mode" not in st.session_state:
        st.session_state["previous_rewrite_mode"] = "ameliorer"

    # Etat interne pour le "pré-check GEO" (binaire et conservateur)
    # - Si on est *certain* que le texte est déjà GEO-friendly : on propose de NE PAS reformuler.
    # - Au moindre doute : on laisse la reformulation se faire normalement.
    if "geo_optimized_block" not in st.session_state:
        st.session_state["geo_optimized_block"] = False
    if "geo_optimized_sig" not in st.session_state:
        st.session_state["geo_optimized_sig"] = ("", "")
    if "geo_skip_notice" not in st.session_state:
        st.session_state["geo_skip_notice"] = False

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
                placeholder="Ex : Titre de page, sous-titre, phrase d’accroche, etc.",
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

        # Si l’utilisateur modifie le texte ou la requête, on invalide le pré-check précédent.
        current_sig = ((original_text or "").strip(), (target_query or "").strip())
        if st.session_state.get("geo_optimized_sig") != current_sig:
            st.session_state["geo_optimized_block"] = False
            st.session_state["geo_skip_notice"] = False

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
                st.session_state["geo_optimized_block"] = False
                st.session_state["geo_skip_notice"] = False

    # Synchroniser la valeur initiale AVANT la création du widget "geo_result_area"
    st.session_state["geo_result_area"] = st.session_state.get(
        "geo_result",
        st.session_state.get("geo_result_area", ""),
    )

    # -------------------------
    # COLONNE DROITE
    # -------------------------
    generate_button_clicked = False
    force_generate_clicked = False

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

            # Message "texte déjà optimisé" (binaire) : on s'arrête ici et on demande confirmation.
            # NB : au moindre doute, on ne bloque pas et on laisse la reformulation se faire.
            if st.session_state.get("geo_skip_notice"):
                st.success("OK — texte conservé (aucune reformulation lancée).")
                st.session_state["geo_skip_notice"] = False

            if (
                st.session_state.get("geo_optimized_block")
                and st.session_state.get("geo_optimized_sig") == current_sig
            ):
                st.warning(
                    "Le texte original est déjà optimisé pour le GEO. Voulez-vous une reformulation quand même ?",
                    icon="🧠",
                )

                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button(
                        "✅ Ne pas reformuler (recommandé)",
                        use_container_width=True,
                        key="geo_skip_rewrite_btn",
                    ):
                        st.session_state["geo_optimized_block"] = False
                        st.session_state["geo_skip_notice"] = True
                        st.rerun()

                with action_cols[1]:
                    force_generate_clicked = st.button(
                        "✍️ Reformuler quand même",
                        use_container_width=True,
                        key="geo_force_rewrite_btn",
                    )

                with st.expander(
                    "Niveaux de réécriture (si vous forcez la reformulation)",
                    expanded=False,
                ):
                    st.markdown(
                        "• **Réécriture minimale** : conserver au maximum le texte d’origine.\n"
                        "• **Améliorer la tournure** : modification du texte d’origine (mêmes idées).\n"
                        "• **Proposition créative** : proposition très différente du texte d’origine (toujours factuelle)."
                    )
                    st.caption("Suggestion : commencez par *Réécriture minimale* pour limiter les changements.")
                    st.caption(
                        "Vous pouvez ajuster le niveau dans la carte \"Niveau de réécriture\" (colonne gauche)."
                    )

            result_text = st.text_area(
                "Résultat",
                height=320,
                key="geo_result_area",
            )
            # Synchronisation avec l'état interne (on ne touche PAS à geo_result_area ici)
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
    if generate_button_clicked or force_generate_clicked:
        if not original_text or not original_text.strip():
            st.warning("Merci de coller un texte à reformuler.")
            return

        if not target_query or not target_query.strip():
            st.warning("Merci de préciser un titre de section ou une requête cible.")
            return

        # Pré-check heuristique (binaire) : on ne bloque QUE si on est certain que le texte est déjà GEO-friendly.
        # Au moindre doute : on laisse la reformulation se faire (fail-open).
        if not force_generate_clicked:
            try:
                already_optimized = geo_is_text_already_optimized(
                    original_text=original_text,
                    target_query=target_query,
                )
            except Exception:
                already_optimized = False

            if already_optimized:
                st.session_state["geo_optimized_block"] = True
                st.session_state["geo_optimized_sig"] = current_sig
                # Suggestion : basculer en réécriture minimale si l’utilisateur force malgré tout.
                st.session_state["geo_rewrite_mode_label"] = "Réécriture minimale"
                st.session_state["previous_rewrite_mode"] = "minimal"
                st.session_state["geo_skip_notice"] = False
                st.rerun()

        # Ici : soit on force la reformulation, soit le texte n'est pas détecté "déjà optimisé".
        st.session_state["geo_optimized_block"] = False

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
                # IMPORTANT : on met uniquement à jour geo_result,
                # PAS geo_result_area (sinon erreur Streamlit).
                st.session_state["geo_result"] = rewritten
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
# ONGLET GEO MONITORING
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
            height=160,
            placeholder="Ex:\nmaison boisset avis\nmaison boisset histoire\nmaison boisset bourgogne",
            key="monitor_queries",
        )
        brand_or_domain = st.text_input(
            "Marque ou domaine à détecter",
            placeholder="Ex : boisset ou boisset.com",
            key="monitor_brand",
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
    queries = [line.strip() for line in (queries_text or "").splitlines() if line.strip()]

    if not queries:
        st.warning("Merci de saisir au moins une requête.")
        return
    if not brand_or_domain.strip():
        st.warning("Merci de saisir une marque ou un domaine à détecter.")
        return

    with st.spinner("Analyse des résultats..."):
        try:
            df = monitor_keywords(
                queries=queries,
                brand_or_domain=brand_or_domain,
                max_results=max_results,
            )
        except Exception as exc:
            st.error(f"Erreur pendant le monitoring : {exc}")
            return

    if df.empty:
        st.info("Aucun résultat récupéré (scraping bloqué ou aucun résultat).")
        return

    hits = int(df["brand_present"].sum())
    total = int(len(df))
    st.metric("Mentions détectées", f"{hits}/{total}")

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
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("GEO Architect – Assistant de reformulation GEO")

    # Diagnostics LLM (seulement en DEV)
    render_backend_diagnostics()

    tab1, tab2 = st.tabs(["🧠 GEO Reformulation", "📊 GEO Monitoring"])

    with tab1:
        render_geo_reformulation_tab()
    with tab2:
        render_geo_monitoring_tab()


if __name__ == "__main__":
    main()
