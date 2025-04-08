# SpotifyPlaylistGenerator 

A web application that creates personalized Spotify playlists based on your mood by analyzing the emotional content of text input. The app bridges natural language processing with Spotify's recommendation algorithms to deliver customized music experiences that match your current emotional state.

## Features

- **Emotion Detection**: Uses advanced NLP models to analyze and classify emotions from text
- **Personalized Recommendations**: Generates playlists customized to both your mood and music preferences
- **Spotify Integration**: Seamlessly creates playlists directly in your Spotify account
- **Multi-strategy Approach**: Implements several fallback mechanisms to ensure quality recommendations
- **User Profile Analysis**: Displays insights about your listening habits and preferences
- **Multilingual Support**: Automatically translates non-English inputs for emotion analysis
- **User-friendly Interface**: Simple and intuitive web interface for all interactions

## How It Works

1. **Text Analysis**: Your input text is processed by a RoBERTa-based emotion detection model
2. **Mood Mapping**: Detected emotions are translated into musical attributes (valence, energy, tempo, etc.)
3. **Music Selection**: The app queries Spotify's API with these attributes and your preferences
4. **Playlist Creation**: A new playlist is automatically created in your Spotify account

## Getting Started

### Prerequisites

- Python 3.7 or higher (Personally used 3.12)
- Spotify Developer account
- Spotify Premium account (for optimal functionality)

### Installation

1. Clone the repository
```bash
git clone https://github.com/Gabrielonee/SpotifyPlaylistGenerator.git
cd SpotifyPlaylistGenerator
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `variables.env` file with your Spotify API credentials:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=decide_yout_redirect_uri
```

4. Run the application
```bash
python main.py
```

5. Navigate to `http://localhost:5001` in your web browser

## Technical Architecture
Architecture of the app has to be upgraded...
### Components

- **Flask Web Server**: Handles HTTP requests and serves the web interface
- **MoodAnalyzer**: Processes text input and extracts emotional content
- **SpotifyMoodAnalyzer**: Converts emotional analysis into music attributes and generates recommendations
- **Spotify OAuth**: Manages authentication with Spotify's API

### Libraries Used

- `flask`: Web application framework
- `spotipy`: Python client for the Spotify Web API
- `transformers`: Hugging Face's NLP library for emotion detection
- `pandas`: Data manipulation and analysis
- `deep_translator`: Text translation for multilingual support
- `numpy`: Numerical operations

## Emotion-to-Music Mapping

The application maps detected emotions to musical attributes:

| Emotion | Valence | Energy | Other Attributes |
|---------|---------|--------|------------------|
| Joy | High | High | High danceability |
| Sadness | Low | Low | Low danceability |
| Anger | Low | Very high | Fast tempo |
| Fear | Medium-low | Medium | High acousticness |
| Surprise | Medium | Medium-high | High loudness |
| Love | Very high | Medium | High acousticness |
| Optimism | High | Medium-high | High danceability |

## Recommendation Strategies

The system employs multiple strategies to find the best music matches:

1. **Genre-based**: Uses emotion-to-genre mapping
2. **User History**: Leverages your top tracks 
3. **Artist-based**: Utilizes your favorite artists
4. **Popularity-based**: Searches popular tracks in relevant genres
5. **Playlist-based**: Samples from public playlists related to the detected mood

## Project Structure

```
SpotifyPlaylistGenerator/
├── main.py               # Flask application and routes
├── sentiment.py          # Emotion analysis and Spotify integration
├── templates/            # HTML templates for web interface
│   ├── index.html        # Landing page
│   ├── user_recap.html   # User profile overview
│   └── recommendations.html  # Displays recommendations
```
## Acknowledgements

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- [Hugging Face Transformers](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)
- [Spotipy](https://spotipy.readthedocs.io/)
- [Flask](https://flask.palletsprojects.com/)

## Contributor
**Soranno Gabriele**
