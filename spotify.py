def getSongData(recommendations, genre):
    song_data_list = []
    for track in recommendations['tracks']:
        # Basic track information
        track_id = track['id']
        duration_ms = track['duration_ms']
        popularity = track['popularity']

        # Collecting genres from all artists of the track
        artist_genres = []
        for artist in track['artists']:
            if 'genres' in artist:
                artist_genres.extend(artist['genres'])

        # Construct the song data object
        song_data = {
            "track_id": track_id,
            "duration_ms": duration_ms,
            "popularity": popularity,
            "artist_genres": [genre] # artist_genres TODO make this list unique - right now always empty
        }
        song_data_list.append(song_data)
    return song_data_list