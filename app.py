import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from scipy.spatial.distance import cdist
import difflib

def getAPIrequest(auth_token, url):
	# To place GET requests to the Spotify API.
    response = requests.get(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {auth_token}"
            }
        )
    return response

def postAPIrequest(auth_token, url, data):
	# To place POST requests to the Spotify API.
    response = requests.post(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {auth_token}"
            }
    )
    return response

def getLastPlayedSongs(numOfTracks):
	# To get the most recently played numOfTracks songs. 
    url = f"https://api.spotify.com/v1/me/player/recently-played?limit={numOfTracks}"
    response = getAPIrequest(auth_token, url)
    response_json = response.json()
    songs = []
    try:
        for song in response_json["items"]:
            songs.append(song)
    except KeyError:
        print("Your Spotify Access Token expired.")
        print("Please obtain a new one and try again.")
        sys.exit(1)
    return songs

def get_song_info(song_list):
	# To get the relevant song information for a list of songs.
    seeds = []
    for item in range(len(song_list)):
        song = {'name': song_list[item]['track']['name'], 'artists': str([song_list[item]['track']['artists'][0]['name']]) }
        seeds.append(song)
    return seeds

def get_song_data(song, song_data):    
	# To match a seeded song with a record in the database. 
    try:
        song_info = song_data[(song_data['name'] == song['name']) 
                            & (song_data['artists'] == song['artists'])].iloc[0]
        return song_info
    except IndexError:
        return None

def get_mean_vector(song_list, song_data):
    # Gets the mean vector for a list of songs.
    song_vectors = []
    for song in song_list:
        song_info = get_song_data(song, song_data)
        if song_info is None:
            print('Warning: {} does not exist in database'.format(song['name']))
            continue
        song_vector = song_info[number_cols].values
        song_vectors.append(song_vector)  
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    # Utility function for flattening a list of dictionaries.
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list, song_data, n_songs=12):
	# Recommends songs based on a list of previous songs that a user has listened to.
    metadata_cols = ['name', 'year', 'artists', 'id']

    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, song_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(song_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = song_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')[1:]

def createPlaylist(name, user_id):
	# Creates a playlist in the user's Spotify account. 
    data = json.dumps({
            "name": name,
            "description": "We hope you enjoy the playlist we curated for you!",
            "public": True
        })
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    response = postAPIrequest(auth_token, url, data)
    response_json = response.json()
    playlist_id = response_json["id"]
    return playlist_id


def searchForTrack(track):
    # Maps a recommended song with its Spotify Web API URI.
    url = f"https://api.spotify.com/v1/tracks/{track['id']}"
    response = getAPIrequest(auth_token, url)
    response_json = response.json()
    track_uri = response_json["uri"]
    return track_uri

def addSongsToPlaylist(playlist_id, tracks):
	# Populates a playlist with a set of songs. 
    track_uris = [searchForTrack(track) for track in tracks]
    data = json.dumps(track_uris)
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    response = postAPIrequest(auth_token, url, data)
    response_json = response.json()
    return response_json

# Set up global variables
auth_token = os.environ.get("SPOTIFY_AUTHORIZATION_TOKEN") 
user_id = os.getenv("SPOTIFY_USER_ID")
if auth_token is None: 
	print("Authorization token is None. Please restart the application after setting up the token.")

# Retreive n most recently played songs
num = int(input("How many tracks would you like to visualize? "))
lastPlayed = getLastPlayedSongs(num)
print(f"\nHere are the last {num} tracks you listened to on Spotify:")
for index, track in enumerate(lastPlayed):
    print(f"\n {index+1}: {track['track']['name']}, {track['track']['artists'][0]['name']} ({track['track']['album']['release_date'][:4]})")
print("\n")

# Allow user to specify the songs they would like
ref_tracks = input("\nEnter a space-separated list of indices to be used as seed tracks (limit: 5): ") 
ref_tracks = ref_tracks.split()
seed_tracks = [lastPlayed[int(i)-1] for i in ref_tracks]
print("\n")

# Building the K-Means Cluster pipeline
song_data = pd.read_csv('./data/data.csv')
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                ('kmeans', KMeans(n_clusters=20, 
                                verbose=2, n_jobs=4))],verbose=True)
X = song_data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
song_data['cluster_label'] = song_cluster_labels
print("\n")
print("\n")

# Making recommendations
recommended = recommend_songs(get_song_info(seed_tracks), song_data)

# Creating a playlist
playlist_name = input("Enter a playlist name: ")
playlist_description = "We hope you enjoy the music we curated for you!"
success = addSongsToPlaylist(createPlaylist(playlist_name, user_id), recommended)
print("\n")
print("\n")
if success: print("Successfully created a playlist. Please check your Spotify!")
print("\n")
print("\n")
