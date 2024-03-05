#!/usr/bin/env python
# coding: utf-8

# # Explore here

# In[1]:


import pandas as pd

tmdb_5000_movies= pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
tmdb_5000_credits= pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv')


# In[2]:


tmdb_5000_movies.info()


# In[3]:


tmdb_5000_credits.info()


# In[4]:


import sqlite3
conn = sqlite3.connect('peliculas.db')
tmdb_5000_movies.to_sql('tmdb_5000_movies', conn, index=False, if_exists='replace')
tmdb_5000_credits.to_sql('tmdb_5000_credits', conn, index=False, if_exists='replace')

query = '''
    SELECT tmdb_5000_credits.movie_id, tmdb_5000_movies.title AS title, tmdb_5000_movies.overview, tmdb_5000_movies.genres, tmdb_5000_movies.keywords, tmdb_5000_credits.cast, tmdb_5000_credits.crew
    FROM tmdb_5000_movies
    JOIN tmdb_5000_credits ON tmdb_5000_movies.title = tmdb_5000_credits.title;
'''

total_data = pd.read_sql(query, conn)

conn.close()

total_data


# In[5]:


import pandas as pd
import json
from pandas import json_normalize

#Seleccionamos las columnas

# Función load_json_safe:
# Esta función toma una cadena JSON (json_str) y devuelve su representación de Python mediante json.loads().
# Se incluye un manejo seguro de errores utilizando un bloque try-except para evitar problemas con cadenas no válidas de JSON.

def load_json_safe(json_str, default_value = None):
    try:
        return json.loads(json_str)
    except (TypeError, json.JSONDecodeError):
        return default_value

# Transformación de la columna "genres" y "keywords":
# Se utiliza la función apply de Pandas en la columna "genres" para aplicar una transformación a cada elemento de la columna.
# Se usa json.loads para convertir la cadena JSON en una lista de diccionarios y luego se extraen los nombres de cada elemento.
# Se verifica si el valor no es nulo (pd.notna(x)) y, en caso contrario, se asigna None.

total_data["genres"] = total_data["genres"].apply(lambda x: [item["name"] for item in json.loads(x)] if pd.notna(x) else None)
total_data["keywords"] = total_data["keywords"].apply(lambda x: [item["name"] for item in json.loads(x)] if pd.notna(x) else None)
# Transformación de la columna "cast":
# Selecciona a los tres primeros actores utilizando la sintaxis de list slicing.
total_data["cast"] = total_data["cast"].apply(lambda x: [item["name"] for item in json.loads(x)][:3] if pd.notna(x) else None)
total_data["crew"] = total_data["crew"].apply(lambda x: " ".join([crew_member['name'] for crew_member in load_json_safe(x) if crew_member['job'] == 'Director']))
total_data["crew"] = total_data["crew"].apply(lambda x: [str(crew_member) for crew_member in x])
total_data["overview"] = total_data["overview"].apply(lambda x: [x])
total_data["crew"] = total_data["crew"].apply(lambda x: [str(crew_member) for crew_member in x])
total_data["crew"] = total_data["crew"].apply(lambda x: ",".join(x).replace(",", ""))


total_data.head()


# In[6]:


total_data["crew"] = total_data["crew"].apply(lambda x: " ".join(x).replace(" ", ""))
total_data["cast"] = [', '.join([nombre.replace(" ", "") for nombre in actor]) for actor in total_data["cast"]]
total_data["genres"] = [', '.join([gen.replace(" ", "") for gen in genre]) for genre in total_data["genres"]]
total_data["keywords"] = [', '.join([word.replace(" ", "") for word in keywords]) for keywords in total_data["keywords"]]
total_data.head()


# In[7]:


total_data["tags"] = total_data["overview"].astype(str) + total_data["genres"].astype(str) + total_data["keywords"].astype(str) + total_data["cast"].astype(str) + total_data["crew"].astype(str)
total_data.drop(columns = ["overview", "genres", "keywords", "cast", "crew"], inplace = True)
total_data.head()


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(total_data["tags"])



# In[9]:


total_data.to_csv("peliculas.csv", index = False)


# In[10]:


from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(n_neighbors = 6, algorithm = "brute", metric = "cosine")
model.fit(tfidf_matrix)


# In[11]:


def recommend(movie):
    movie_index = total_data[total_data["title"] == movie].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_index])
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    similar_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
    return similar_movies[1:]

input_movie = "Avatar"
recommendations = recommend(input_movie)
print("Film recommendations '{}'".format(input_movie))
for movie, distance in recommendations:
    print("- Film: {}".format(movie))


# In[12]:


from pickle import dump

dump(model, open("./model.pkl", "wb"))
dump(model, open("./model.sav", "wb"))

