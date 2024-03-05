import streamlit as st
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import scipy as sp

def load_data():
    game_df = pd.read_csv("steam-200k.csv")
    game_df = game_df.rename(columns={"The Elder Scrolls V Skyrim": "games",
                                      "151603712": "User_ID",
                                      "1.0": "Hoursplay",
                                      "purchase": "Status"})
  
    game_df = game_df.drop(columns=['0'])
    game_df = game_df.drop_duplicates(['User_ID', 'games'], keep='last')
    game_df.dropna(inplace=True)

    game_df = game_df[(game_df['Hoursplay'] >= 2) & (game_df['Status'] == 'play')]
    game_filtered_df = game_df[game_df.groupby('games').User_ID.transform(len) >= 10]
    game_filtered_df['User_ID'] = game_filtered_df['User_ID'].astype(str)
    
    averages = game_filtered_df.groupby(['games'], as_index=False).Hoursplay.mean()
    averages['avg_Hoursplay'] = averages['Hoursplay']
    averages.drop('Hoursplay', axis=1, inplace=True)
    
    final_ratings = pd.merge(game_filtered_df, averages[['games', 'avg_Hoursplay']], on='games')
    conditions = [(final_ratings['Hoursplay'] >= 0.8 * final_ratings['avg_Hoursplay']),
                  (final_ratings['Hoursplay'] >= 0.6 * final_ratings['avg_Hoursplay']) &(final_ratings['Hoursplay'] < 0.8 * final_ratings['avg_Hoursplay']),
                  (final_ratings['Hoursplay'] >= 0.4 * final_ratings['avg_Hoursplay']) &(final_ratings['Hoursplay'] < 0.6 * final_ratings['avg_Hoursplay']),
                  (final_ratings['Hoursplay'] >= 0.2 * final_ratings['avg_Hoursplay']) &(final_ratings['Hoursplay'] < 0.4 * final_ratings['avg_Hoursplay']),
                  final_ratings['Hoursplay'] >= 0]
    values = [5, 4, 3, 2, 1]
    final_ratings['rating'] = np.select(conditions, values)
    final_ratings = final_ratings.drop(['Status', 'Hoursplay', 'avg_Hoursplay'], axis=1)
    return final_ratings

final_ratings = load_data()

piv = final_ratings.pivot_table(index=['User_ID'], columns=['games'], values='rating')

piv_norm = piv.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=1)
piv_norm.fillna(0, inplace=True)
piv_norm = piv_norm.T
piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]

# pretvaranje podataka u format matrice
piv_sparse = sp.sparse.csr_matrix(piv_norm.values)
item_similarity = cosine_similarity(piv_sparse)
user_similarity = cosine_similarity(piv_sparse.T)

item_sim_df = pd.DataFrame(item_similarity, index=piv_norm.index, columns=piv_norm.index)
user_sim_df = pd.DataFrame(user_similarity, index=piv_norm.columns, columns=piv_norm.columns)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(piv_norm)

def matching_score(a,b):
   return fuzz.ratio(a,b)

def get_title_from_index(index):
   return final_ratings[final_ratings.index == index]['games'].values[0]

def find_closest_title(title):
   leven_scores = list(enumerate(final_ratings['games'].apply(matching_score, b=title)))
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
   closest_title = get_title_from_index(sorted_leven_scores[0][0])
   distance_score = sorted_leven_scores[0][1]
   return closest_title, distance_score

def main():
    st.title("Collaborative Filtering Recommendation System")

    input_game = st.text_input("Enter the game which you want to be recommended by:")
    if st.button("Get recommendations"):
        closest_title, distance_score = find_closest_title(input_game)

        if closest_title in piv_norm.index:
            query_index = piv_norm.index.get_loc(closest_title)
            distances, indices = model_knn.kneighbors(piv_norm.iloc[query_index, :].values.reshape(1, -1), n_neighbors=11)
            st.text(f"Closest Title: {closest_title}")
            st.text("Users who like " + closest_title + " also like: ")
            for i in range(1, len(distances.flatten())):
                st.text(f"{i}: {piv_norm.index[indices.flatten()[i]]}, recommended based on the ratings of users who have played both games, with a similarity in ratings of {int(distances.flatten()[i]*100)}%")
        else:
            st.text(f"The game '{closest_title}' is not found in the dataset.")

if __name__ == "__main__":
    main()


            

