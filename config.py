import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key')
    SESSION_COOKIE_NAME = 'mood_music_session'
    PERMANENT_SESSION_LIFETIME = 3600
    CACHE_EXPIRY_DAYS = 7
    FAMILIAR_PROPORTION = 0.6
    SIMILARITY_THRESHOLD = 0.7