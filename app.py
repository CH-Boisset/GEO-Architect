import json

import streamlit as st
import streamlit.components.v1 as components

from config import (
    GEO_ENV,
    IS_PROD,
    DEFAULT_GEMINI_MODEL,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    DEFAULT_BACKEND,
)
from geo_utils import (
    geo_rewrite_content,
    monitor_keywords,
    test_gemini_connection,
    test_ollama_connection,
)


st.set_page_config(
    page_title="GEO Architect - Assistant MVP GEO",
    layout="wide",
)


def render_backend_diagnostics():
    """
    Petit bloc de diagnostic pour vérifier rapidement les backends LLM.
    """
    with st.expander("🔍 Diagnostics des backends LLM"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Ollama (local)")
            st.code(f"Base URL : {OLLAMA_BASE_URL}\nModèle : {OLLAMA_MODEL_NAME}")
            if st.button("🧪 Tester Ollama", key="test_ollama"):
                msg = test_ollama_connection()
                if "OK" in msg:
                    st.success(msg)
                else:
                    st.error(msg)

        with col2:
            st.markdown("#### Gemini (cloud)")
            st.write("Le test utilisera la clé API définie en .env ou celle saisie ci-dessous.")
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


def render_geo_reformulation_tab():
    """Affiche l’onglet de reformulation GEO (titre de section + texte + backend + modes)."""
    st.header("🧠 GEO Reformulation")
    st.write(
        """
        Colle un contenu existant, renseigne un titre de section (ou une requête cible simple),
        puis laisse l'assistant générer une version optimisée GEO : neutre, factuelle,
        structurée, prête à être exploitée par des moteurs IA.
        """
    )

    if IS_PROD:
        st.info(
            "Cette version de GEO Architect utilise l’API Gemini de Google pour la reformulation. "
            "Ne colle pas de données strictement confidentielles ou sensibles. "
            "Limite-toi aux contenus destinés à être publiés (sites web, plaquettes, contenus marketing)."
        )

    if not IS_PROD:
        st.markdown(f"*Environnement actuel* : `{GEO_ENV}`")
    else:
        st.caption("Environnement : production – backend Gemini 2.5-flash.")

    # Backend Selection
    col_backend, col_mode = st.columns(2)
    with col_backend:
        if IS_PROD:
            backend = "gemini"
            st.caption("Backend IA : Gemini (forcé en production).")
        else:
            backend_choice = st.radio(
                "Moteur IA (Backend)",
                ["Ollama (Local)", "Gemini (Cloud)"],
                index=0 if DEFAULT_BACKEND == "ollama" else 1,
                horizontal=True,
            )
            backend = "ollama" if "Ollama" in backend_choice else "gemini"

        if "previous_backend" not in st.session_state:
            st.session_state["previous_backend"] = backend
        elif st.session_state["previous_backend"] != backend:
            st.session_state["geo_result"] = ""
            st.session_state["geo_result_area"] = ""
            st.session_state["previous_backend"] = backend

    with col_mode:
        # Mode de réécriture (remplace le slider de température)
        mode_label_to_value = {
            "Réécriture minimale": "minimal",
            "Améliorer la tournure": "ameliorer",
            "Proposition créative": "creatif",
        }
        mode_label = st.selectbox(
            "Niveau de réécriture",
            list(mode_label_to_value.keys()),
            index=1,  # "ameliorer" par défaut
        )
        rewrite_mode = mode_label_to_value[mode_label]

        st.caption(
            "• Réécriture minimale : corrections et ajustements très légers.  "
            "• Améliorer la tournure : texte plus fluide, même contenu.  "
            "• Proposition créative : style plus travaillé, toujours factuel."
        )

    if backend == "ollama":
        st.caption(f"Modèle local : `{OLLAMA_MODEL_NAME}` sur `{OLLAMA_BASE_URL}`")
    else:
        if IS_PROD:
            st.caption(f"Modèle IA : `{DEFAULT_GEMINI_MODEL}` (Gemini)")
        else:
            st.caption(f"Modèle cloud : `{DEFAULT_GEMINI_MODEL}` (nécessite une clé API)")

    # Reset result if rewrite mode changes
    if "previous_rewrite_mode" not in st.session_state:
        st.session_state["previous_rewrite_mode"] = rewrite_mode
    elif st.session_state["previous_rewrite_mode"] != rewrite_mode:
        st.session_state["geo_result"] = ""
        st.session_state["geo_result_area"] = ""
        st.session_state["previous_rewrite_mode"] = rewrite_mode


    st.markdown("### Contenu à optimiser")

    col_left, col_right = st.columns(2)

    with col_left:
        target_query = st.text_input(
            "Titre de section (ou requête cible simple)",
            placeholder="Ex : Notre héritage, maison boisset histoire...",
        )
        original_text = st.text_area(
            "Texte original",
            height=400,
            placeholder="Colle ici le texte à reformuler dans une logique GEO...",
        )

        generate_button = st.button(
            "🚀 Générer la version GEO",
            type="primary",
        )

    # Logique de génération (AVANT l'affichage du résultat)
    if generate_button:
        if not original_text.strip():
            st.warning("Merci de coller un texte à reformuler.")
            return
        if not target_query.strip():
            st.warning("Merci de préciser un titre de section (ou une requête cible principale).")
            return

        with st.spinner(f"Génération de la version GEO en cours ({backend})..."):
            try:
                rewritten = geo_rewrite_content(
                    original_text=original_text,
                    target_query=target_query,
                    model_name=None,  # Laisse geo_utils choisir le défaut
                    rewrite_mode=rewrite_mode,
                    backend=backend,
                    user_api_key=None,  # La clé GEMINI_API_KEY est gérée côté serveur (.env)
                )
                # On met à jour le state ET le widget key pour être sûr
                st.session_state["geo_result"] = rewritten
                st.session_state["geo_result_area"] = rewritten
                st.rerun()

            except Exception as exc:
                st.error(f"Erreur lors de la reformulation : {exc}")

    with col_right:
        st.markdown("**Texte GEO optimisé**")
        
        # Init widget key if needed to avoid "default value" warning
        if "geo_result_area" not in st.session_state:
            st.session_state["geo_result_area"] = st.session_state.get("geo_result", "")

        result_text = st.text_area(
            "Résultat",
            height=400,
            key="geo_result_area",
        )
        # Sync inverse : si l'utilisateur édite, on met à jour le state
        st.session_state["geo_result"] = result_text

        if result_text:
            col_success, col_copy = st.columns([3, 1])
            with col_success:
                st.success("✅ Texte optimisé généré avec succès !")
            with col_copy:
                # Bouton HTML/JS pur pour garantir l'accès au presse-papiers côté client.
                # Le texte copié est exactement celui affiché (y compris après édition manuelle).
                safe_text = json.dumps(result_text)

                components.html(
                    f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <style>
                        body {{ margin: 0; padding: 0; font-family: sans-serif; }}
                        .copy-btn {{
                            background-color: #f0f2f6;
                            border: 1px solid #d6d9df;
                            border-radius: 4px;
                            color: #31333F;
                            padding: 0.5rem 1rem;
                            font-size: 1rem;
                            cursor: pointer;
                            width: 100%;
                            transition: all 0.2s;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 8px;
                        }}
                        .copy-btn:hover {{
                            border-color: #ff4b4b;
                            color: #ff4b4b;
                        }}
                        .copy-btn:active {{
                            background-color: #ff4b4b;
                            color: white;
                        }}
                        .copy-btn.copied {{
                            background-color: #d1fae5;
                            border-color: #10b981;
                            color: #065f46;
                        }}
                    </style>
                    </head>
                    <body>
                        <button class="copy-btn" onclick="copyText(this)">
                            <span data-icon>📋</span> <span data-label>Copier le texte</span>
                        </button>

                        <script>
                        const text = {safe_text};

                        const setState = (btn, state) => {{
                            const icon = btn.querySelector('[data-icon]');
                            const label = btn.querySelector('[data-label]');
                            if (state === 'success') {{
                                btn.classList.add('copied');
                                icon.textContent = '✓';
                                label.textContent = 'Copié !';
                                setTimeout(() => setState(btn, 'idle'), 2000);
                            }} else {{
                                btn.classList.remove('copied');
                                icon.textContent = '📋';
                                label.textContent = 'Copier le texte';
                            }}
                        }};

                        async function copyText(btn) {{
                            try {{
                                if (navigator.clipboard && window.isSecureContext) {{
                                    await navigator.clipboard.writeText(text);
                                }} else {{
                                    throw new Error('Clipboard API indisponible');
                                }}
                                setState(btn, 'success');
                                return;
                            }} catch (err) {{
                                console.warn('Clipboard API échec, fallback execCommand', err);
                            }}

                            try {{
                                const textarea = document.createElement('textarea');
                                textarea.value = text;
                                textarea.setAttribute('readonly', '');
                                textarea.style.position = 'fixed';
                                textarea.style.top = '-1000px';
                                document.body.appendChild(textarea);
                                textarea.select();
                                const ok = document.execCommand('copy');
                                document.body.removeChild(textarea);
                                if (!ok) {{
                                    throw new Error('execCommand a retourné false');
                                }}
                                setState(btn, 'success');
                            }} catch (fallbackErr) {{
                                console.error('Erreur copie (fallback):', fallbackErr);
                                setState(btn, 'idle');
                                alert('Erreur lors de la copie. Essayez Ctrl+A / Ctrl+C manuellement.');
                            }}
                        }}
                        </script>
                    </body>
                    </html>
                    """,
                    height=50,  # Hauteur suffisante pour le bouton
                )



def render_geo_monitoring_tab():
    """Affiche l’onglet de monitoring (analyse de présence sur mots-clés)."""
    st.header("📊 GEO Monitoring (simple)")

    st.write(
        """
        Fournis une ou plusieurs requêtes et une marque / un domaine.
        L'application interroge DuckDuckGo (scraping léger) et indique si la marque
        apparaît dans les premiers résultats.
        """
    )

    st.caption(
        "⚠️ Ce monitoring est volontairement simple et limité. "
        "Il ne doit pas être utilisé pour du scraping massif (risque de blocage / "
        "non-respect des conditions d'utilisation des moteurs)."
    )

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

    if submitted:
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


def main():
    st.title("GEO Architect – Assistant MVP GEO")

    # En mode DEV uniquement, on affiche les diagnostics LLM
    if not IS_PROD:
        render_backend_diagnostics()
        st.warning("Environnement : DEV – ne pas utiliser en production.")
    else:
        # En PROD, simple message d'information
        st.info(
            "Cette version de GEO Architect utilise l’API Gemini en mode cloud. "
            "Ne collez pas de données sensibles ou strictement confidentielles."
        )

    tab1, tab2 = st.tabs(["🧠 GEO Reformulation", "📊 GEO Monitoring"])

    with tab1:
        render_geo_reformulation_tab()
    with tab2:
        render_geo_monitoring_tab()


if __name__ == "__main__":
    main()
