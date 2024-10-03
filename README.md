# Spotify ML

## Summary
This project explores the possibility of audio features having an impact on a songs popularity. We gather these features from Spotify and store them in a MongoDB.
We then split the Data to use if for ML training/testing.

## Structure

### Common
This package includes all common code that is shared among multiple programs/packages.

### Data Gathering
This package contains all the code for calling the spotify API to gather all necessary Song Data for our Machine Learning and stores it in a Database.

### ML Training
This package contains all the code for our machine learning algorithms which take data from the database and make the popularity predications.

## How to start the project
Change the Spotify Client ID in the spotify.py to your own from https://developer.spotify.com/dashboard
Add SPOTIFY_SECRET as a environment variable
