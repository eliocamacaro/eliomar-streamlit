import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#Cargar modelo
with open('model.pkl' ,'rb') as md:
    model = pickle.load(md)
#Cargar dataset
total_data = pd.read_csv('peliculas.csv')

#Vectorizer para leer el index de peliculas
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(total_data["tags"])


def main():
    st.title("Recomendaciones de peliculas")

    st.sidebar.header("Selecciona tu pelicula")

    def recommend(movie):
        movie_index = total_data[total_data["title"] == movie].index[0]
        distances, indices = model.kneighbors(tfidf_matrix[movie_index])
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        similar_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
        return similar_movies[1:]


    movie_list = total_data["title"].to_list()
    input_movie = st.sidebar.selectbox("Indica tu pelicula", movie_list)
    recommendations = recommend(input_movie)

    button = st.sidebar.button("Mostrar")

    st.subheader("Tus recomendaciones aqui:")

    if button:
        st.subheader(input_movie)
        st.write("Film recommendations '{}'".format(input_movie))
        for movie, distance in recommendations:
            st.write("- Film: {}".format(movie))
    else:
        st.write("Aqui apareceran las recomendaciones")

if __name__ == "__main__":
    main()
