import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Use your Spotify API credentials instead
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Fetch user's recently played tracks
results = sp.current_user_recently_played()

# Extract relevant information from the API response
tracks = []
for item in results['items']:
    track = item['track']
    tracks.append({
        'track_name': track['name'],
        'artist_name': track['artists'][0]['name'],
        'album_name': track['album']['name'],
        'popularity': track['popularity'],
        'uri': track['uri']
    })

# Create a DataFrame from the collected data
df = pd.DataFrame(tracks)

# Feature Engineering
df = pd.get_dummies(df, columns=['artist_name'])
scaler = MinMaxScaler()
df['popularity'] = scaler.fit_transform(df[['popularity']])

# Building the Recommendation Model
X = df.drop(['track_names', 'album_names', 'urll'], axis=1)
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(X)

# User Preferences
user_preferences = {
    'popularity': 0.8,
    'artist_name_TaylorSwift': 1,
    'artist_name_Ed Sheeran': 0
}

# Transform user preferences
user_features = [user_preferences[feature] if feature in user_preferences else 0 for feature in X.columns]

# Find similar tracks
_, indices = model.kneighbors([user_features])
recommended_tracks = df.loc[indices[0]]['uri']

# Display Recommendations
for track_uri in recommended_tracks:
    track_info = sp.track(track_uri)
    print(f"Advisable Song track: {track_info['name']} by {track_info['artists'][0]['name']}")
