import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from app.config import Config
from app.models.user import User


def get_auth_url():
    global _spotify_service_instance
    if _spotify_service_instance is None:
        _spotify_service_instance = SpotifyService()
    sp_oauth = _spotify_service_instance.get_oauth_client()
    return sp_oauth.get_authorize_url()

def process_callback(code):
    global _spotify_service_instance
    if _spotify_service_instance is None:
        _spotify_service_instance = SpotifyService()
    sp_oauth = _spotify_service_instance.get_oauth_client()
    try:
        token_info = sp_oauth.get_access_token(code)
        if token_info:
            return {'success': True, 'token_info': token_info}
        else:
            return {'success': False, 'error': 'Token non ottenuto'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


class SpotifyService:
    def __init__(self):
        self.client_id = Config.SPOTIFY_CLIENT_ID
        self.client_secret = Config.SPOTIFY_CLIENT_SECRET
        self.redirect_uri = Config.SPOTIFY_REDIRECT_URI
        
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ),
            requests_timeout=20
        )
_spotify_service_instance = None

def get_spotify_client():
    global _spotify_service_instance
    if _spotify_service_instance is None:
        _spotify_service_instance = SpotifyService()
    return _spotify_service_instance.authenticate_user()


class SpotifyService:
    def __init__(self):
        self.client_id = Config.SPOTIFY_CLIENT_ID
        self.client_secret = Config.SPOTIFY_CLIENT_SECRET
        self.redirect_uri = Config.SPOTIFY_REDIRECT_URI
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ),
            requests_timeout=20
        )

    def get_oauth_client(self):
        return SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join([
                'user-library-read',
                'user-top-read',
                'user-read-recently-played',
                'playlist-read-private',
                'playlist-modify-private',
                'playlist-modify-public'
            ]),
            cache_path=".spotify_cache",
            show_dialog=True
        )

    def authenticate_user(self):
        sp_oauth = self.get_oauth_client()
        token_info = sp_oauth.get_cached_token()
        if not token_info or sp_oauth.is_token_expired(token_info):
            return None
        else:
            return spotipy.Spotify(auth=token_info['access_token'])

    def get_user_data(self, sp_user):
        user_profile = sp_user.current_user()
        user = User(user_profile)
        user.top_tracks = {
            'short_term': sp_user.current_user_top_tracks(time_range='short_term', limit=20),
            'medium_term': sp_user.current_user_top_tracks(time_range='medium_term', limit=30),
            'long_term': sp_user.current_user_top_tracks(time_range='long_term', limit=40)
        }
        user.top_artists = {
            'short_term': sp_user.current_user_top_artists(time_range='short_term', limit=20),
            'medium_term': sp_user.current_user_top_artists(time_range='medium_term', limit=30),
            'long_term': sp_user.current_user_top_artists(time_range='long_term', limit=40)
        }
        user.recently_played = sp_user.current_user_recently_played(limit=30)
        import pandas as pd
        genres = []
        for artist in user.top_artists['medium_term']['items']:
            genres.extend(artist['genres'])
        user.top_genres = pd.Series(genres).value_counts().to_dict()
        return user