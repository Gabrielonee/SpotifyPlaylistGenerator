

The app will be available at [http://localhost:5001](http://localhost:5001)
# Spotify Playlist Generator

## Project Overview
Spotify Playlist Generator is a web application that analyzes the user's mood or emotions from a text input and generates a personalized Spotify playlist that matches the detected mood. The app leverages the Spotify Web API for user authentication, music data retrieval, and playlist creation, combining it with sentiment analysis and custom recommendation logic.

## Features
- **Spotify OAuth Authentication**: Secure login with Spotify to access user data.
- **Mood Analysis**: Analyzes user input text to detect emotions using NLP techniques.
- **Personalized Recommendations**: Suggests tracks, artists, and genres based on the user's mood and listening history.
- **Playlist Creation**: Automatically creates a playlist on the user's Spotify account with the recommended tracks.
- **User Recap**: Displays user's top tracks, artists, genres, and recently played songs.

## Project Structure
```
SpotifyPlaylistGenerator/
│   main.py                  # Entry point for the Flask app
│   variables.env            # Environment variables (not tracked by git)
│
├── app/
│   ├── __init__.py
│   ├── config.py            # App configuration
│   ├── controllers/         # Flask controllers (routes logic)
│   │   ├── auth_controller.py
│   │   ├── music_controller.py
│   │   └── routes.py
│   ├── models/              # Data models (User, Emotion, etc.)
│   │   ├── user.py
│   │   └── emotion.py
│   ├── services/            # Business logic and Spotify API integration
│   │   ├── spotify_services.py
│   │   ├── recommendation.py
│   │   └── mood_analysis.py
│   ├── utils/               # Utility modules (cache, translation, etc.)
│   ├── static/              # Static files (CSS, JS, images)
│   └── templates/           # HTML templates (Jinja2)
│       ├── index.html
│       ├── recommendations.html
│       ├── user_recap.html
│       └── error.html
└── .gitignore               # Git ignore rules
```

## How It Works
1. **User Authentication**: The user logs in with Spotify. The app retrieves the user's profile and listening data.
2. **Mood Input**: The user describes their current mood or situation in a text box.
3. **Emotion Detection**: The app analyzes the text to extract emotions (e.g., joy, sadness, anger).
4. **Music Recommendation**: Based on the detected emotions and user history, the app selects suitable tracks and artists.
5. **Playlist Creation**: The app creates a new playlist on the user's Spotify account and adds the recommended tracks.
6. **Recap & Results**: The user can view their music recap and open the generated playlist directly in Spotify.

## Requirements
- Python 3.8+
- Flask
- spotipy
- Other dependencies listed in `requirements.txt`

## Run with Docker

You can run the application in a containerized environment using Docker. Make sure you have a `variables.env` file with your Spotify credentials in the project root.

### Build the Docker image
```sh
docker build -t spotify-playlist-generator .
```

### Run the container
```sh
docker run --env-file variables.env -p 5001:5001 spotify-playlist-generator
```

Or with Docker Compose:
```sh
docker-compose up --build
```

## Setup & Run
1. Clone the repository.
2. Create a `.env`file with your Spotify API credentials.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python main.py`
5. Open your browser at `http://localhost:5001`

## Notes
- Do not commit your `.env` or `variables.env` files. (use a gitignro file)

## License
MIT License
