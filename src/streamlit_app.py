"""
streamlit_app.py
================
Sistema de Recomendación de Películas - MovieLens 100K
"""

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIGURACIÓN =====
st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide"
)

# ===== ESTILOS =====
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    color: #E50914;
    text-align: center;
    font-weight: bold;
}
.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===== CARGAR DATOS =====
@st.cache_data
def load_data():
    DB_PATH = 'movielens.db'     
    conn = sqlite3.connect(DB_PATH)
    
    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
    
    users = pd.read_sql("SELECT * FROM users", conn)
    users.columns = ['userId', 'age', 'gender', 'occupation', 'zipcode']
    
    movies = pd.read_sql("SELECT * FROM items", conn)
    movies.rename(columns={'item_id': 'movieId'}, inplace=True)
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    conn.close()
    
    df = ratings.merge(users, on='userId').merge(movies, on='movieId')
    
    return df, movies

df, movies_df = load_data()

# ===== PREPARAR DATOS PARA RECOMENDACIONES =====
@st.cache_data
def prepare_data(_df, _movies):
    # Popularidad
    popular = _df.groupby(['movieId', 'title']).agg({
        'rating': ['mean', 'count']
    }).reset_index()
    popular.columns = ['movieId', 'title', 'avg_rating', 'num_ratings']
    popular = popular[popular['num_ratings'] >= 20]
    popular = popular.sort_values('avg_rating', ascending=False)
    
    # Content-Based
    genre_cols = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 
                  'Thriller', 'War', 'Western']
    
    movies_unique = _movies[['movieId', 'title', 'year'] + genre_cols].drop_duplicates()
    genre_matrix = movies_unique[genre_cols].values
    similarity = cosine_similarity(genre_matrix)
    
    # Collaborative
    user_movie = _df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_movie)
    
    return popular, movies_unique, similarity, user_movie, user_similarity

popular, movies_unique, similarity, user_movie, user_similarity = prepare_data(df, movies_df)

# ===== HEADER =====
st.markdown('<div class="main-title">Sistema de Recomendación de Películas</div>', 
            unsafe_allow_html=True)
st.markdown("### MovieLens 100K Dataset")

# ===== MÉTRICAS =====
col1, col2, col3, col4 = st.columns(4)
col1.metric("Películas", f"{df['movieId'].nunique():,}")
col2.metric("Usuarios", f"{df['userId'].nunique():,}")
col3.metric("Ratings", f"{len(df):,}")
col4.metric("Rating Promedio", f"{df['rating'].mean():.2f}/5")

st.divider()

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["Populares", "Similares", "Para Ti", "Exploración"])

# ===== TAB 1: POPULARES =====
with tab1:
    st.subheader("Top Películas Más Populares")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Número de películas:", 5, 20, 10)
    with col2:
        min_ratings = st.slider("Mínimo valoraciones:", 10, 100, 30)
    
    filtered = popular[popular['num_ratings'] >= min_ratings].head(n)
    
    for i, row in filtered.iterrows():
        st.write(f"**{i+1}. {row['title']}**")
        st.write(f"Rating: {row['avg_rating']:.2f}/5 • Valoraciones: {int(row['num_ratings'])}")
        st.divider()

# ===== TAB 2: SIMILARES =====
with tab2:
    st.subheader("Encuentra Películas Similares")
    
    movie_list = sorted(movies_unique['title'].unique()[:300])
    selected = st.selectbox("Selecciona una película:", movie_list)
    n_similar = st.slider("Número de recomendaciones:", 5, 15, 10, key="similar_n")
    
    if st.button("Buscar Similares", type="primary"):
        matches = movies_unique[movies_unique['title'] == selected]
        
        if len(matches) > 0:
            idx = matches.index[0]
            sim_scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:n_similar+1]
            
            indices = [i[0] for i in sim_scores]
            scores = [i[1] for i in sim_scores]
            
            result = movies_unique.iloc[indices][['title', 'year']].copy()
            result['similarity'] = [f"{s*100:.1f}%" for s in scores]
            
            st.success(f"Películas similares a: **{selected}**")
            st.dataframe(result, width='stretch', hide_index=True)

# ===== TAB 3: PARA TI =====
with tab3:
    st.subheader("Recomendaciones Personalizadas")
    
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.number_input("User ID (1-943):", 1, 943, 1)
    with col2:
        n_recs = st.slider("Recomendaciones:", 5, 15, 10, key="collab_n")
    
    if st.button("Ver Recomendaciones", type="primary"):
        if user_id in user_movie.index:
            idx = list(user_movie.index).index(user_id)
            sim_scores = sorted(enumerate(user_similarity[idx]), key=lambda x: x[1], reverse=True)[1:11]
            
            similar_users = [i[0] for i in sim_scores]
            recommendations = user_movie.iloc[similar_users].mean(axis=0)
            user_ratings = user_movie.loc[user_id]
            recommendations = recommendations[user_ratings == 0]
            top = recommendations.nlargest(n_recs)
            
            result = pd.DataFrame({'movieId': top.index, 'predicted': top.values})
            result = result.merge(movies_df[['movieId', 'title', 'year']], on='movieId')
            result['predicted'] = result['predicted'].round(2)
            
            st.success(f"Recomendaciones para Usuario #{user_id}")
            st.dataframe(result[['title', 'year', 'predicted']], width='stretch', hide_index=True)
        else:
            st.error("Usuario no encontrado")

# ===== TAB 4: EXPLORACIÓN =====
with tab4:
    st.subheader("Exploración de Datos")
    
    # Distribución ratings
    st.write("**Distribución de Ratings**")
    rating_dist = df['rating'].value_counts().sort_index()
    fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                labels={'x': 'Rating', 'y': 'Cantidad'},
                color_discrete_sequence=['#E50914'])
    st.plotly_chart(fig, width='stretch')
    
    # Géneros
    st.write("**Géneros Más Populares**")
    genre_cols = ['Action', 'Comedy', 'Drama', 'Sci_Fi', 'Thriller']
    genre_counts = df[genre_cols].sum().sort_values(ascending=False)
    fig2 = px.bar(x=genre_counts.index, y=genre_counts.values,
                 labels={'x': 'Género', 'y': 'Películas'},
                 color_discrete_sequence=['#E50914'])
    st.plotly_chart(fig2, width='stretch')

# ===== FOOTER =====
st.divider()
st.markdown("**Sistema de Recomendación MovieLens 100K** ")