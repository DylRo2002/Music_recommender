import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    df = load_and_preprocess_data("data/song-dataset.csv")
    df_similarities = compute_song_similarities(df)

    while True:
        print("Best 10 recommendations")
        print("---------------------------------")
        print("10 songs similar to yours.")
        
        input_song = get_input_song(df_similarities)
        if input_song is None:
            break

        recommendation = get_song_recommendation(df_similarities, input_song)
        print("new songs to check:")
        for song in recommendation:
            print(song)
        
        next_command = input("Generate again for the next song? [yes, no] ")
        if next_command.lower() == "no":
            break

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)[:1000]
    df = df.drop_duplicates(subset="Song Name").dropna(axis=0)
    df = df.drop(df.columns[3:], axis=1)
    df["Artist Name"] = df["Artist Name"].str.replace(" ", "")
    df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)
    return df

def compute_song_similarities(df):
    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(df["data"])
    similarities = cosine_similarity(vectorized)
    df_similarities = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()
    return df_similarities

def get_input_song(df_similarities):
    while True:
        input_song = input("Please enter the name of the song: ")

        if input_song in df_similarities.columns:
            return input_song
        else:
            print("Song not found in the database. Please try again or type 'no' to exit.")
            exit_choice = input("Exit? [yes, no] ")
            if exit_choice.lower() == "yes":
                return None

def get_song_recommendation(df_similarities, input_song):
    recommendation = df_similarities.nlargest(11, input_song)["Song Name"]
    return recommendation.values[1:]

if _name_ == "_main_":
    main()
