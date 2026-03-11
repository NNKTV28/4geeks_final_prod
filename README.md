<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=MovieLens%20100K%20Recommender&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=32" width="100%" />

  <a href="https://github.com/NNKTV28/4geeks_final_prod">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=1000&center=true&vCenter=true&multiline=true&width=700&height=80&lines=Sistema+de+Recomendaci%C3%B3n+de+Pel%C3%ADculas+%F0%9F%8E%AC;User-CF+%7C+Item-CF+%7C+SVD+%E2%80%94+MovieLens+100K+%F0%9F%93%8A" alt="Typing SVG" />
  </a>

  <br/>

  <img src="https://komarev.com/ghpvc/?username=NNKtv28&label=Repo%20Views&color=0e75b6&style=flat" alt="Repo Views" />
  &nbsp;
  <a href="https://github.com/NNKTV28/4geeks_final_prod"><img src="https://img.shields.io/github/stars/NNKTV28/4geeks_final_prod?style=flat&color=yellow" /></a>
</div>

---

## 🎥 Sobre el Proyecto

Siempre nos pareció interesante cómo plataformas como Netflix o Spotify saben qué recomendarte. Así que para nuestro proyecto final del bootcamp de Data Science en **4Geeks Academy**, decidimos construir nuestro propio **sistema de recomendación de películas** desde cero usando el dataset **MovieLens 100K**.

La idea fue simple: tomar datos reales de preferencias de usuarios, entenderlos a fondo y probar distintos enfoques de Machine Learning para ver cuál recomienda mejor. Fue un viaje de aprendizaje donde tocamos desde la limpieza de datos hasta la evaluación de modelos.

---

## 💾 Los Datos

### ¿De dónde salen?

Usamos el dataset **MovieLens 100K**, uno de los más conocidos en el mundo de los sistemas de recomendación. Fue recopilado por el [GroupLens Research Project](https://grouplens.org/) de la Universidad de Minnesota a través de su plataforma web durante 7 meses (septiembre 1997 – abril 1998). Es un dataset clásico que se sigue usando hoy en día como benchmark.

### ¿Qué contiene?

| Dato | Cantidad |
|------|----------|
| Ratings (1–5) | **100,000** |
| Usuarios | **943** |
| Películas | **1,682** |
| Info demográfica | edad, género, ocupación, código postal |

- Cada usuario valoró al menos 20 películas, así que no hay usuarios "fantasma".
- GroupLens ya limpió los datos por nosotros: eliminaron usuarios con pocos ratings o sin info demográfica completa.

### Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `u.data` | 100K ratings completos (user, item, rating, timestamp) |
| `u.user` | Info demográfica de usuarios |
| `u.item` | Info de películas + 19 flags binarios de género |
| `ua.base` / `ua.test` | Split de entrenamiento/test predefinido |
| `u.genre` | Lista de géneros |
| `u.occupation` | Lista de ocupaciones |

---

## ⚙️ Procesamiento

### ¿Cómo tratamos los datos?

El dataset original viene en archivos de texto plano separados por pipes (`|`) y tabs. Lo primero que hicimos fue pasarlo todo a una base de datos SQLite para poder trabajar más cómodo con SQL y Pandas.

```
data/raw/ml-100k/ ──→ load_to_sqlite.py ──→ data/movielens.db ──→ EDA & Modelos
```

1. **Carga a SQLite**: Escribimos un script (`src/load_to_sqlite.py`) que parsea todos los archivos y crea tablas limpias: `users`, `items`, `ratings`, `genres`, `occupations` y los splits de train/test.

2. **Unificación**: Juntamos ratings + usuarios + películas en un solo DataFrame para tener toda la info a mano.

3. **Limpieza**: Revisamos nulos (casi no había, solo en campos opcionales como `video_release_date`) y duplicados (0%). También extrajimos el año de lanzamiento del título con regex.

4. **Feature Engineering**: Creamos grupos de edad, agrupamos películas por década, y construimos la **matriz usuario-ítem** (943 × 1682). La centramos restando la media de cada usuario para capturar mejor si a alguien le gustó una peli *más o menos que su promedio*.

<div align="center">

### Tech Stack

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=plotly&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

</div>

---

## 📊 EDA — ¿Qué nos dicen los datos?

Antes de meternos con los modelos, nos tomamos el tiempo de explorar bien los datos. Algunos hallazgos interesantes:

- **La gente tiende a puntuar alto** → La mayoría de ratings son 3 o 4 estrellas. Pocos usuarios se molestan en dar un 1.
- **El usuario típico** → Hombre, entre 20-35 años, estudiante. Hay un sesgo demográfico claro.
- **Las pelis viejas puntúan mejor** → Las películas de los 90s dominan en cantidad, pero los clásicos de décadas anteriores tienen mejores ratings. Esto no es que fueran mejores — es un *sesgo de supervivencia* (solo los clásicos reconocidos están en el catálogo).
- **Drama es el rey** → Es el género con más películas y más ratings. Pero los géneros de nicho como Film-Noir y War tienen los promedios más altos.
- **Los géneros se juntan** → Drama + Romance, Action + Adventure, Thriller + Crime son combos frecuentes.
- **La edad casi no importa** → La correlación entre edad del usuario, año de la película y rating es prácticamente nula.

---

## 🤖 Machine Learning — ¿Qué modelos probamos?

Probamos **3 enfoques distintos** de recomendación para ver cuál funcionaba mejor. Usamos el split predefinido del dataset (`ua_base` para entrenar, `ua_test` para evaluar):

### Los 3 modelos

| # | Modelo | ¿Qué hace? |
|---|--------|-------------|
| 1 | **User-Based CF** | Busca usuarios parecidos a ti y te recomienda lo que les gustó a ellos (k=30 vecinos) |
| 2 | **Item-Based CF** | Busca películas parecidas a las que ya te gustaron (k=30 vecinos) |
| 3 | **SVD** | Descompone la matriz de ratings para encontrar "gustos ocultos" de cada usuario (k=50 factores) |

### Resultados

| Sistema | RMSE ↓ | MAE ↓ |
|---------|--------|-------|
| User-Based CF | ~0.99 | ~0.78 |
| Item-Based CF | ~1.03 | ~0.82 |
| **SVD (k=50)** | **~0.95** | **~0.75** |

> 🏆 **Ganador: SVD** — Se equivoca menos que los otros dos. Tiene sentido: en vez de buscar vecinos uno a uno, captura patrones globales de todo el dataset.

### También medimos...

- **Precision@K** y **Recall@K** (con K = 5, 10, 15, 20) para ver si las recomendaciones top son realmente relevantes.
- **Distribución de errores** → Los 3 modelos se equivocan de forma simétrica (no hay sesgo sistemático), pero SVD tiene menos dispersión.
- **Radar chart** que compara todo de un vistazo: RMSE, MAE, Precision@10 y Recall@10.

---

## ⚠️ Limitaciones (siendo honestos)

- **Es un dataset de los 90s**: 100K ratings suenan como mucho, pero hoy Netflix maneja miles de millones. Los gustos de 1997 no son los de hoy.
- **El problema del "usuario nuevo"**: Si llega alguien sin historial, los modelos no saben qué recomendarle. Es el clásico *cold-start*.
- **Muchos huecos**: La matriz usuario-ítem tiene un ~93.7% de celdas vacías. La mayoría de usuarios solo vieron una fracción pequeña del catálogo.
- **Solo usamos ratings**: No incorporamos sinopsis, directores, actores ni posters. Con esa info las recomendaciones podrían mejorar bastante.
- **Evaluación estática**: Medimos todo sobre un split fijo. En la vida real habría que probar con usuarios de verdad (tests A/B).
- **Sesgo del dataset**: Mayoritariamente hombres jóvenes angloparlantes. Las recomendaciones pueden no generalizar bien a otros perfiles.

---

## 💡 ¿Qué nos gustaría agregar?

- 🚀 **Probar Deep Learning**: Modelos como Neural Collaborative Filtering o autoencoders que podrían superar a SVD.
- 🔍 **Mezclar enfoques**: Combinar lo que hacemos (filtrado colaborativo) con análisis del contenido de las pelis (sinopsis, actores, directores).
- 🌐 **Usar más datos**: MovieLens 25M tiene 250 veces más ratings. Sería interesante ver cómo escalan nuestros modelos.
- 👥 **Resolver el cold-start**: Recomendar por popularidad o usar info demográfica cuando no hay historial.

---

## 📁 Estructura del Proyecto

```
4geeks_final_prod/
├── data/
│   ├── raw/
│   │   └── ml-100k/          # Dataset original MovieLens 100K
│   ├── interim/               # Datos intermedios
│   └── processed/             # Datos procesados
├── src/
│   ├── app.py                 # Script principal
│   ├── explore.ipynb          # Notebook de exploración y EDA
│   ├── load_to_sqlite.py      # Carga de datos a SQLite
│   ├── utils.py               # Funciones auxiliares
│   └── eda_nikita/
│       └── evaluate_models.py # EDA completo + evaluación de 3 modelos
├── models/                    # Modelos entrenados
├── requirements.txt           # Dependencias Python
└── README.md
```

---

## 🚀 Instalación y Uso

```bash
# Clonar el repositorio
git clone https://github.com/NNKTV28/4geeks_final_prod.git
cd 4geeks_final_prod

# Instalar dependencias
pip install -r requirements.txt

# Cargar datos a SQLite
python src/load_to_sqlite.py

# Ejecutar EDA + evaluación de modelos
python src/eda_nikita/evaluate_models.py

# URL STREAMLIT
https://movielens-recommender-b49e.onrender.com/
```

---

## 👥 Contributors

<div align="center">

| Contributor | GitHub |
|-------------|--------|
| **NNKtv28** | [![GitHub](https://img.shields.io/badge/-NNKtv28-181717?style=flat&logo=github&logoColor=white)](https://github.com/NNKtv28) |
| **Paloma Gondim Pereira** | [![GitHub](https://img.shields.io/badge/-palomagondim-181717?style=flat&logo=github&logoColor=white)](https://github.com/palomagondim) |
| **4Geeks Academy** | [![GitHub](https://img.shields.io/badge/-4GeeksAcademy-181717?style=flat&logo=github&logoColor=white)](https://github.com/4GeeksAcademy) |

</div>

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%" />