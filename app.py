import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Segoe UI Emoji', 'Segoe UI', 'sans-serif']
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title="Explorateur de Profils · K-Means",
    page_icon="🔍",
    layout="wide"
)

# ── Styles CSS personnalisés ──────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0f1e; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #00d2ff !important; }
    h2, h3 { color: #e2e8f0 !important; }
    .metric-card {
        background: #0f1829;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .stMetric label { color: #64748b !important; font-size: 12px !important; }
    .stMetric value { color: #00d2ff !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════


@st.cache_data
def generate_dataset(seed=42):
    """Génère 75 profils étudiants synthétiques en 3 groupes naturels."""
    np.random.seed(seed)

    # Groupe 1 — Studieux
    g1 = pd.DataFrame({
        'temps_etude':        np.random.normal(16, 2, 25),
        'note_moyenne':       np.random.normal(17, 1, 25),
        'participation':      np.random.normal(8.5, 0.8, 25),
        'nb_absences':        np.random.normal(1, 0.7, 25),
        'heures_coding':      np.random.normal(18, 2, 25),
        'preference_theorie': np.random.normal(8, 1, 25),
    })
    # Groupe 2 — Équilibrés
    g2 = pd.DataFrame({
        'temps_etude':        np.random.normal(9, 2, 25),
        'note_moyenne':       np.random.normal(12, 1.5, 25),
        'participation':      np.random.normal(5.5, 1, 25),
        'nb_absences':        np.random.normal(5, 1.5, 25),
        'heures_coding':      np.random.normal(8, 2, 25),
        'preference_theorie': np.random.normal(5, 1, 25),
    })
    # Groupe 3 — En difficulté
    g3 = pd.DataFrame({
        'temps_etude':        np.random.normal(3, 1.5, 25),
        'note_moyenne':       np.random.normal(7, 1.5, 25),
        'participation':      np.random.normal(2, 1, 25),
        'nb_absences':        np.random.normal(11, 2, 25),
        'heures_coding':      np.random.normal(2, 1, 25),
        'preference_theorie': np.random.normal(3, 1, 25),
    })

    df = pd.concat([g1, g2, g3], ignore_index=True)
    df = df.clip(lower=0)
    df['note_moyenne'] = df['note_moyenne'].clip(0, 20)
    return df.round(2)


# ══════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#03060f,#0a1628);
            border:1px solid #00d2ff33;border-radius:14px;
            padding:28px 36px;margin-bottom:24px">
  <div style="font-size:12px;color:#4a5568;letter-spacing:4px;margin-bottom:8px">
    MINI-PROJET IA A9 · APPRENTISSAGE NON SUPERVISÉ
  </div>
  <div style="font-size:28px;font-weight:800;
              background:linear-gradient(90deg,#00d2ff,#9d50bb);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Explorateur de Profils par Clustering K-Means
  </div>
  <div style="color:#4a5568;font-size:13px;margin-top:8px">
    Proposé par Ing/Dr. Ghaith Khlifi &nbsp;·&nbsp;
    Réalisé par les deux collègues: Refka Barhoumi &nbsp;·&nbsp; Ons Ahmadi
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — PARAMÈTRES
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    # Slider k
    k = st.slider(
        "Nombre de clusters (k)",
        min_value=2, max_value=6, value=3, step=1,
        help="Changer k pour voir l'effet sur les clusters"
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    n_total = st.slider("Nombre de profils", 30, 150, 75, 15)
    seed = st.number_input("Graine aléatoire", value=42, step=1)

    st.markdown("---")
    st.markdown("### 🎨 Affichage")
    show_centroids = st.checkbox("Afficher les centroïdes", value=True)
    show_table = st.checkbox("Afficher le tableau", value=True)

    st.markdown("---")
    st.caption("Refka Barhoumi · Mini-projet IA A9")

# ══════════════════════════════════════════════════════════════
# TRAITEMENT
# ══════════════════════════════════════════════════════════════
df = generate_dataset(int(seed))
# Ajuster la taille si besoin
if n_total != 75:
    np.random.seed(int(seed))
    idx = np.random.choice(len(df), min(n_total, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-Means
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df['cluster'] = labels

# PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centres_pca = pca.transform(km.cluster_centers_)

# Noms automatiques des clusters selon note_moyenne
centres_orig = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=df.columns[:-1]
)
order = centres_orig['note_moyenne'].rank(ascending=False).astype(int)
noms_base = {1: "🏆 Studieux", 2: "🎯 Équilibrés", 3: "🔄 En difficulté",
             4: "📉 Très faibles", 5: "🌟 Excellence", 6: "📊 Groupe 6"}
cluster_names = {i: noms_base.get(order[i], f"Groupe {i}") for i in range(k)}
df['profil'] = df['cluster'].map(cluster_names)

COLORS = ['#00d2ff', '#ff2d78', '#00ff87', '#ffd700', '#9d50bb', '#ff6b35']

# ══════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📋 Profils", len(df))
with col2:
    st.metric("🔢 Clusters (k)", k)
with col3:
    st.metric("📉 Inertie (WCSS)", f"{km.inertia_:.0f}")
with col4:
    st.metric("📐 PCA variance",
              f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# GRAPHIQUES PRINCIPAUX
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔵 Scatter PCA", "📊 Bar Chart", "🌡️ Heatmap", "📈 Elbow Method"
])

# ── TAB 1 : Scatter PCA ───────────────────────────────────────
with tab1:
    st.markdown("#### Visualisation des clusters avec PCA")
    st.caption(
        f"PCA réduit les 6 dimensions en 2 · Variance expliquée : {sum(pca.explained_variance_ratio_)*100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0f1829')

    for ci in range(k):
        mask = labels == ci
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[ci], label=cluster_names[ci],
                   alpha=0.75, s=70, edgecolors='white', linewidths=0.3)

    if show_centroids:
        ax.scatter(centres_pca[:, 0], centres_pca[:, 1],
                   c='white', marker='X', s=220, zorder=10,
                   label='Centroïdes', edgecolors='black', linewidths=0.5)

    ax.set_xlabel(f"Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                  color='#64748b')
    ax.set_ylabel(f"Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                  color='#64748b')
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.legend(facecolor='#0f1829', edgecolor='#1e3a5f', labelcolor='white')
    ax.grid(True, alpha=0.1, color='white')

    st.pyplot(fig)
    plt.close()

# ── TAB 2 : Bar Chart ─────────────────────────────────────────
with tab2:
    st.markdown("#### Moyennes des attributs par cluster")

    moyennes = df.groupby('profil')[df.columns[:6]].mean().round(2)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0f1829')

    x = np.arange(6)
    width = 0.8 / k
    attrs = ['temps_etude', 'note_moyenne', 'participation',
             'nb_absences', 'heures_coding', 'preference_theorie']
    labels_fr = ['Étude', 'Note', 'Partic.', 'Absences', 'Coding', 'Théorie']

    for ci, (nom, color) in enumerate(zip(moyennes.index, COLORS)):
        bars = ax.bar(x + ci * width, moyennes.loc[nom, attrs],
                      width * 0.9, label=nom, color=color, alpha=0.85)

    ax.set_xticks(x + width * (k - 1) / 2)
    ax.set_xticklabels(labels_fr, color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#64748b')
    ax.legend(facecolor='#0f1829', edgecolor='#1e3a5f', labelcolor='white')
    ax.grid(True, alpha=0.1, axis='y', color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')

    st.pyplot(fig)
    plt.close()

# ── TAB 3 : Heatmap ───────────────────────────────────────────
with tab3:
    st.markdown("#### Profil moyen de chaque cluster")

    moyennes2 = df.groupby('cluster')[df.columns[:6]].mean().round(2)
    moyennes2.index = [cluster_names[i] for i in moyennes2.index]

    fig, ax = plt.subplots(figsize=(10, max(3, k * 1.2)))
    fig.patch.set_facecolor('#0a0f1e')

    sns.heatmap(moyennes2, annot=True, fmt='.1f',
                cmap='YlOrRd', ax=ax,
                linewidths=0.5, linecolor='#0a0f1e',
                cbar_kws={'label': 'Valeur moyenne'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                       ha='right', color='#94a3b8')
    ax.set_yticklabels(ax.get_yticklabels(), color='#94a3b8')

    st.pyplot(fig)
    plt.close()

# ── TAB 4 : Elbow Method ─────────────────────────────────────
with tab4:
    st.markdown("#### Méthode Elbow — Choisir k optimal")
    st.caption(
        "Le coude de la courbe indique le meilleur compromis entre précision et complexité")

    wcss = []
    k_range = range(1, 9)
    for ki in k_range:
        km_temp = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km_temp.fit(X_scaled)
        wcss.append(km_temp.inertia_)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0f1829')

    ax.plot(list(k_range), wcss, 'o-', color='#00d2ff',
            linewidth=2.5, markersize=8, markerfacecolor='#00d2ff')
    ax.axvline(x=k, color='#ff2d78', linestyle='--',
               linewidth=2, label=f'k={k} (sélectionné)')
    ax.scatter([k], [wcss[k-1]], color='#ff2d78', s=180, zorder=5)

    ax.set_xlabel("Nombre de clusters k", color='#64748b')
    ax.set_ylabel("WCSS (Inertie)", color='#64748b')
    ax.set_title("Courbe Elbow", color='#e2e8f0', pad=12)
    ax.tick_params(colors='#64748b')
    ax.legend(facecolor='#0f1829', edgecolor='#1e3a5f', labelcolor='white')
    ax.grid(True, alpha=0.1, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')

    st.pyplot(fig)
    plt.close()

    st.info(
        f"💡 WCSS pour k={k} : **{wcss[k-1]:.0f}** · Le coude optimal se situe généralement à k=3")

# ══════════════════════════════════════════════════════════════
# TABLEAU DES PROFILS
# ══════════════════════════════════════════════════════════════
if show_table:
    st.markdown("---")
    st.markdown("#### 📋 Tableau des profils avec cluster assigné")

    col_filter, col_search = st.columns([2, 3])
    with col_filter:
        filtre = st.selectbox("Filtrer par profil",
                              ["Tous"] + list(df['profil'].unique()))
    with col_search:
        st.caption(f"Total: {len(df)} profils · {k} clusters")

    df_display = df if filtre == "Tous" else df[df['profil'] == filtre]

    st.dataframe(
        df_display[['temps_etude', 'note_moyenne', 'participation',
                    'nb_absences', 'heures_coding', 'profil']].reset_index(drop=True),
        use_container_width=True,
        height=300
    )

# ══════════════════════════════════════════════════════════════
# INTERPRÉTATION
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 🧠 Interprétation des clusters")

cols = st.columns(k)
for ci in range(k):
    with cols[ci]:
        nom = cluster_names[ci]
        cnt = (df['cluster'] == ci).sum()
        moy = centres_orig.iloc[ci]
        pct = cnt / len(df) * 100

        st.markdown(f"""
        <div style="background:#0f1829;border:1px solid {COLORS[ci]}44;
                    border-left:3px solid {COLORS[ci]};
                    border-radius:10px;padding:16px">
          <div style="color:{COLORS[ci]};font-size:16px;font-weight:700;margin-bottom:8px">
            {nom}
          </div>
          <div style="color:#64748b;font-size:12px;margin-bottom:10px">
            {cnt} profils ({pct:.0f}%)
          </div>
          <div style="color:#94a3b8;font-size:13px;line-height:1.7">
            📚 Étude: <b>{moy['temps_etude']:.1f}h/sem</b><br>
            🎓 Note: <b>{moy['note_moyenne']:.1f}/20</b><br>
            💻 Coding: <b>{moy['heures_coding']:.1f}h/sem</b><br>
            ❌ Absences: <b>{moy['nb_absences']:.1f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2d3748;font-size:12px;padding:10px">
  Mini-Projet IA A9 · K-Means Clustering · Refka Barhoumi ·
  Proposé par Ing/Dr. Ghaith Khlifi
</div>
""", unsafe_allow_html=True)
