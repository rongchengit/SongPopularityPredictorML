# Function to add a song if it doesn't exist in the database
def add_song_if_not_exists(collection, song_data):
    try: 
        for song in song_data:
            # Check if the song exists in the database
            if collection.find_one({"track_id": song["track_id"]}) is None:
                # Song doesn't exist, so add it
                collection.insert_one(song)
                print(f"Song added: {song['track_id']}")
            else:
                # Song already exists
                print(f"Song already exists: {song['track_id']}")
    except Exception as e:
        print(f"Error Storing Song in Database: {e}")