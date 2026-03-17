import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

DATABASE_PATH = "movielens.db"
BRAND_COLOR = "#E50914"
MIN_RATINGS_DEFAULT = 20
MIN_RATINGS_SLIDER_MIN = 10
MIN_RATINGS_SLIDER_MAX = 100
MIN_RATINGS_SLIDER_DEFAULT = 30
MOVIES_DISPLAY_MIN = 5
MOVIES_DISPLAY_MAX = 20
MOVIES_DISPLAY_DEFAULT = 10
RECOMMENDATIONS_MIN = 5
RECOMMENDATIONS_MAX = 15
RECOMMENDATIONS_DEFAULT = 10
SELECTABLE_MOVIE_LIMIT = 300
SIMILAR_USERS_COUNT = 10
USER_ID_MIN = 1
USER_ID_MAX = 943

ALL_GENRE_COLUMNS = [
    "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
    "Thriller", "War", "Western",
]

EXPLORATION_GENRE_COLUMNS = ["Action", "Comedy", "Drama", "Sci_Fi", "Thriller"]

CUSTOM_STYLES = """
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
"""

def configurePageLayout():
    st.set_page_config(
        page_title="MovieLens Recommender",
        page_icon="🎬",
        layout="wide",
    )
    st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)


@st.cache_data
def loadRatingsFromDatabase():
    connection = sqlite3.connect(DATABASE_PATH)
    ratings = pd.read_sql("SELECT * FROM ratings", connection)
    ratings.columns = ["userId", "movieId", "rating", "timestamp"]
    connection.close()
    return ratings


@st.cache_data
def loadUsersFromDatabase():
    connection = sqlite3.connect(DATABASE_PATH)
    users = pd.read_sql("SELECT * FROM users", connection)
    users.columns = ["userId", "age", "gender", "occupation", "zipcode"]
    connection.close()
    return users


@st.cache_data
def loadMoviesFromDatabase():
    connection = sqlite3.connect(DATABASE_PATH)
    movies = pd.read_sql("SELECT * FROM items", connection)
    movies.rename(columns={"item_id": "movieId"}, inplace=True)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    connection.close()
    return movies


@st.cache_data
def buildMergedDataset(ratings, users, movies):
    return ratings.merge(users, on="userId").merge(movies, on="movieId")


@st.cache_data
def buildPopularityRanking(mergedDataset):
    popularity = mergedDataset.groupby(["movieId", "title"]).agg(
        {"rating": ["mean", "count"]}
    ).reset_index()
    popularity.columns = ["movieId", "title", "averageRating", "totalRatings"]
    popularity = popularity[popularity["totalRatings"] >= MIN_RATINGS_DEFAULT]
    return popularity.sort_values("averageRating", ascending=False)


@st.cache_data
def buildUniqueMoviesWithGenres(movies):
    selectedColumns = ["movieId", "title", "year"] + ALL_GENRE_COLUMNS
    return movies[selectedColumns].drop_duplicates()


@st.cache_data
def computeGenreSimilarityMatrix(uniqueMovies):
    genreMatrix = uniqueMovies[ALL_GENRE_COLUMNS].values
    return cosine_similarity(genreMatrix)


@st.cache_data
def buildUserMovieMatrix(mergedDataset):
    return mergedDataset.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)


@st.cache_data
def computeUserSimilarityMatrix(userMovieMatrix):
    return cosine_similarity(userMovieMatrix)


def findSimilarMovies(selectedTitle, uniqueMovies, similarityMatrix, numberOfResults):
    matchingMovies = uniqueMovies[uniqueMovies["title"] == selectedTitle]

    if matchingMovies.empty:
        return pd.DataFrame()

    movieIndex = matchingMovies.index[0]
    similarityScores = sorted(
        enumerate(similarityMatrix[movieIndex]),
        key=lambda pair: pair[1],
        reverse=True,
    )[1 : numberOfResults + 1]

    similarIndices = [pair[0] for pair in similarityScores]
    similarScores = [pair[1] for pair in similarityScores]

    similarMovies = uniqueMovies.iloc[similarIndices][["title", "year"]].copy()
    similarMovies["similarity"] = [f"{score * 100:.1f}%" for score in similarScores]
    return similarMovies


def getCollaborativeRecommendations(
    targetUserId, userMovieMatrix, userSimilarityMatrix, moviesDataframe, numberOfResults
):
    if targetUserId not in userMovieMatrix.index:
        return None

    userIndex = list(userMovieMatrix.index).index(targetUserId)
    similarityScores = sorted(
        enumerate(userSimilarityMatrix[userIndex]),
        key=lambda pair: pair[1],
        reverse=True,
    )[1 : SIMILAR_USERS_COUNT + 1]

    similarUserIndices = [pair[0] for pair in similarityScores]
    averageRatingsFromSimilarUsers = userMovieMatrix.iloc[similarUserIndices].mean(axis=0)

    targetUserRatings = userMovieMatrix.loc[targetUserId]
    unseenMovieScores = averageRatingsFromSimilarUsers[targetUserRatings == 0]
    topRecommendations = unseenMovieScores.nlargest(numberOfResults)

    recommendationResults = pd.DataFrame({
        "movieId": topRecommendations.index,
        "predictedRating": topRecommendations.values,
    })
    recommendationResults = recommendationResults.merge(
        moviesDataframe[["movieId", "title", "year"]], on="movieId"
    )
    recommendationResults["predictedRating"] = recommendationResults["predictedRating"].round(2)
    return recommendationResults


def renderHeader(mergedDataset):
    st.markdown(
        '<div class="main-title">Sistema de Recomendación de Películas</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### MovieLens 100K Dataset")

    metricColumn1, metricColumn2, metricColumn3, metricColumn4 = st.columns(4)
    metricColumn1.metric("Películas", f"{mergedDataset['movieId'].nunique():,}")
    metricColumn2.metric("Usuarios", f"{mergedDataset['userId'].nunique():,}")
    metricColumn3.metric("Ratings", f"{len(mergedDataset):,}")
    metricColumn4.metric("Rating Promedio", f"{mergedDataset['rating'].mean():.2f}/5")

    st.divider()


def renderPopularMoviesTab(popularityRanking):
    st.subheader("Top Películas Más Populares")

    filterColumn1, filterColumn2 = st.columns(2)
    with filterColumn1:
        numberOfMovies = st.slider(
            "Número de películas:",
            MOVIES_DISPLAY_MIN,
            MOVIES_DISPLAY_MAX,
            MOVIES_DISPLAY_DEFAULT,
        )
    with filterColumn2:
        minimumRatingsThreshold = st.slider(
            "Mínimo valoraciones:",
            MIN_RATINGS_SLIDER_MIN,
            MIN_RATINGS_SLIDER_MAX,
            MIN_RATINGS_SLIDER_DEFAULT,
        )

    filteredMovies = popularityRanking[
        popularityRanking["totalRatings"] >= minimumRatingsThreshold
    ].head(numberOfMovies)

    for position, (_, movieRow) in enumerate(filteredMovies.iterrows(), start=1):
        st.write(f"**{position}. {movieRow['title']}**")
        st.write(
            f"Rating: {movieRow['averageRating']:.2f}/5 "
            f"• Valoraciones: {int(movieRow['totalRatings'])}"
        )
        st.divider()


def renderSimilarMoviesTab(uniqueMovies, similarityMatrix):
    st.subheader("Encuentra Películas Similares")

    availableMovieTitles = sorted(uniqueMovies["title"].unique()[:SELECTABLE_MOVIE_LIMIT])
    selectedMovieTitle = st.selectbox("Selecciona una película:", availableMovieTitles)
    numberOfSimilarMovies = st.slider(
        "Número de recomendaciones:",
        RECOMMENDATIONS_MIN,
        RECOMMENDATIONS_MAX,
        RECOMMENDATIONS_DEFAULT,
        key="similar_n",
    )

    if st.button("Buscar Similares", type="primary"):
        similarMovies = findSimilarMovies(
            selectedMovieTitle, uniqueMovies, similarityMatrix, numberOfSimilarMovies
        )

        if not similarMovies.empty:
            st.success(f"Películas similares a: **{selectedMovieTitle}**")
            st.dataframe(similarMovies, width="stretch", hide_index=True)


def renderPersonalizedRecommendationsTab(
    userMovieMatrix, userSimilarityMatrix, moviesDataframe
):
    st.subheader("Recomendaciones Personalizadas")

    inputColumn1, inputColumn2 = st.columns(2)
    with inputColumn1:
        selectedUserId = st.number_input(
            f"User ID ({USER_ID_MIN}-{USER_ID_MAX}):",
            USER_ID_MIN,
            USER_ID_MAX,
            USER_ID_MIN,
        )
    with inputColumn2:
        numberOfRecommendations = st.slider(
            "Recomendaciones:",
            RECOMMENDATIONS_MIN,
            RECOMMENDATIONS_MAX,
            RECOMMENDATIONS_DEFAULT,
            key="collab_n",
        )

    if st.button("Ver Recomendaciones", type="primary"):
        recommendations = getCollaborativeRecommendations(
            selectedUserId,
            userMovieMatrix,
            userSimilarityMatrix,
            moviesDataframe,
            numberOfRecommendations,
        )

        if recommendations is not None:
            st.success(f"Recomendaciones para Usuario #{selectedUserId}")
            st.dataframe(
                recommendations[["title", "year", "predictedRating"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.error("Usuario no encontrado")


def renderExplorationTab(mergedDataset):
    st.subheader("Exploración de Datos")

    st.write("**Distribución de Ratings**")
    ratingDistribution = mergedDataset["rating"].value_counts().sort_index()
    ratingChart = px.bar(
        x=ratingDistribution.index,
        y=ratingDistribution.values,
        labels={"x": "Rating", "y": "Cantidad"},
        color_discrete_sequence=[BRAND_COLOR],
    )
    st.plotly_chart(ratingChart, width="stretch")

    st.write("**Géneros Más Populares**")
    genreTotals = mergedDataset[EXPLORATION_GENRE_COLUMNS].sum().sort_values(ascending=False)
    genreChart = px.bar(
        x=genreTotals.index,
        y=genreTotals.values,
        labels={"x": "Género", "y": "Películas"},
        color_discrete_sequence=[BRAND_COLOR],
    )
    st.plotly_chart(genreChart, width="stretch")


def renderFooter():
    st.divider()
    st.markdown("**Sistema de Recomendación MovieLens 100K** ")


def main():
    configurePageLayout()

    ratings = loadRatingsFromDatabase()
    users = loadUsersFromDatabase()
    movies = loadMoviesFromDatabase()
    mergedDataset = buildMergedDataset(ratings, users, movies)

    popularityRanking = buildPopularityRanking(mergedDataset)
    uniqueMovies = buildUniqueMoviesWithGenres(movies)
    genreSimilarityMatrix = computeGenreSimilarityMatrix(uniqueMovies)
    userMovieMatrix = buildUserMovieMatrix(mergedDataset)
    userSimilarityMatrix = computeUserSimilarityMatrix(userMovieMatrix)

    renderHeader(mergedDataset)

    tabPopular, tabSimilar, tabPersonalized, tabExploration = st.tabs(
        ["Populares", "Similares", "Para Ti", "Exploración"]
    )

    with tabPopular:
        renderPopularMoviesTab(popularityRanking)

    with tabSimilar:
        renderSimilarMoviesTab(uniqueMovies, genreSimilarityMatrix)

    with tabPersonalized:
        renderPersonalizedRecommendationsTab(
            userMovieMatrix, userSimilarityMatrix, movies
        )

    with tabExploration:
        renderExplorationTab(mergedDataset)

    renderFooter()


main()
