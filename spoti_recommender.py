import spotipy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from spotipy.oauth2 import SpotifyClientCredentials

# While we do not have access to spotify we use a previously saved dataset
df = pd.read_csv('songs_data.csv', index_col='Unnamed: 0')
print(df)

# Initialize Spotipy with User Credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='',
                                                           client_secret=''))

# In my account look for the first 50 songs of Justin Bieber that appear in search
results = sp.search(q='artist:Justin Bieber', limit=50) # returns a json to analyze with JSON_lint
print(results)

# Extract the track id
tracks_ids = [tracks['id'] for track in results['tracks']['items']]
audio_features = sp.audio_features(tracks_ids) # the method that returns all features from ids
df = pd.DataFrame(audio_features)

# Create a function to automate the process of extracting audio features
def get_audio_features(artist):
    results = sp.search(q=f'artist:{artist}', limit=50)
    tracks_ids = [track['id'] for track in results['tracks']['items']]
    song_names = [track['name'] for track in results['tracks']['items']]
    # extract
    audio_features = sp.audio_features(track_ids)
    # store features in dataframe
    df = pd.DataFrame(audio_features)
    df['artist'] = artist #
    df['song_name'] = song_name
    return df


# create a loop and select artists you want to get songs from
artists = ['Nirvana', 'Justin Bieber', 'deadmau5']
df = pd.DataFrame()

for artist in artists:
    df_artist = get_audio_features(artist)
    df = pd.concat([df, df_artist])  # overwrites the empty df defined above and appends

df = df.reset_index(drop=True)
print(df)

# NOW lets have fun and fit the KMeans model
x = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
x_prep = StandardScaler().fit_transform(x)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_prep)

clusters = kmeans.predict(x_prep)
df['cluster'] = clusters
print(df)

# Analyze our results to see how the model is behaving (not very reliable)
df.groupby(['cluster', 'artist'], as_index=False).count().sort_values(['cluster', 'song name'],
                                                                      ascending=False)[['cluster', 'artist', 'song name']]

# Analyze a new input song
input_song = input('Choose a song: ')
results = sp.search(q=f'track:{input_song}', limit=1)
track_id = results['tracks']['items'][0]['id']
audio_features = sp.audio_features(track_id)
df_ = pd.DataFrame(audio_features)
x = df_[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# Calculate and output the closest song in the dataset
closest, _ = pairwise_distances_argmin_min(x, df[x.columns])
print('Here is a similar song: {}'.format(str(closest)))

df.loc[closest]['song name'], df.loc[closest]['artist']

def recommend_song():
    song_name = input('Ehy you pick a song!: ')
    results = sp.search(q=f'track:{song_name}', limit=1)
    track_id = results['tracks']['items'][0]['id']
    audio_features = sp.audio_features(track_id)
    df_ = pd.DataFrame(audio_features)
    x = df_[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
             'instrumentalness', 'liveness', 'valence', 'tempo']]
    closest, _ = pairwise_distances_argmin_min(x, df[x.columns])
    return ' - '.join([df.loc[closest]['song name'].values[0], df.loc[closest]['artist'].values[0]])

print(recommend_song())

