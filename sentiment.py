import time
import pandas as pd
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os
from dotenv import load_dotenv

# sentiment detection
from transformers import pipeline
import numpy as np

# Env variables
load_dotenv('variables.env')

class MoodAnalyzer:     
    def __init__(self):
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=None
        )
        
        self.emotion_mapping = {
            'joy': {'target_valence': 0.85, 'target_energy': 0.75, 'target_danceability': 0.8},
            'sadness': {'target_valence': 0.25, 'target_energy': 0.3, 'target_danceability': 0.3},
            'anger': {'target_energy': 0.9, 'target_valence': 0.2, 'target_tempo': 140},
            'fear': {'target_energy': 0.6, 'target_valence': 0.4, 'target_acousticness': 0.8},
            'surprise': {'target_energy': 0.7, 'target_valence': 0.6, 'target_loudness': -5},
            'love': {'target_valence': 0.9, 'target_energy': 0.6, 'target_acousticness': 0.7},
            'optimism': {'target_valence': 0.8, 'target_energy': 0.7, 'target_danceability': 0.75}
        }

    def analyze_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        results = self.emotion_analyzer(text)
        print(f"Risultati del modello: {results}")  # Debug
        
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        
        total = sum(res['score'] for res in results)
        emotions = {
            str(res['label']).lower(): res['score'] / total
            for res in results
        }
        
        return emotions
    
class SpotifyMoodAnalyzer:
    def __init__(self, client_id=None, client_secret=None, redirect_uri=None):
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = redirect_uri or os.getenv('SPOTIFY_REDIRECT_URI')
        
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ),
            requests_timeout=20
        )
    
        self.mood_analyzer = MoodAnalyzer() 

    def analyze_text(self, user_input):
        return self.mood_analyzer.analyze_text(user_input)

    def authenticate_user(self, scope=None):
        if scope is None:
            scope = [
                'user-library-read',
                'user-top-read',
                'user-read-recently-played',
                'playlist-read-private',
                'playlist-modify-private',
                'playlist-modify-public'
            ]
        
        sp_oauth = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join(scope),
            cache_path=".spotify_cache",
            show_dialog=True
        )
        
        token_info = sp_oauth.get_cached_token()
        
        if not token_info or sp_oauth.is_token_expired(token_info):
            return None
        else:
            return spotipy.Spotify(auth=token_info['access_token'])
    
    def get_user_data(self, sp_user):
        user_profile = sp_user.current_user()
        
        top_tracks = {
            'short_term': sp_user.current_user_top_tracks(time_range='short_term', limit=10),
            'medium_term': sp_user.current_user_top_tracks(time_range='medium_term', limit=10),
            'long_term': sp_user.current_user_top_tracks(time_range='long_term', limit=10)
        }
    
        top_artists = {
            'short_term': sp_user.current_user_top_artists(time_range='short_term', limit=10),
            'medium_term': sp_user.current_user_top_artists(time_range='medium_term', limit=10),
            'long_term': sp_user.current_user_top_artists(time_range='long_term', limit=10)
        }
        
        recently_played = sp_user.current_user_recently_played(limit=10)
        
        genres = []
        for artist in top_artists['medium_term']['items']:
            genres.extend(artist['genres'])
        genre_counts = pd.Series(genres).value_counts().to_dict()
        
        return {
            'user_profile': user_profile,
            'top_tracks': top_tracks,
            'top_artists': top_artists,
            'recently_played': recently_played,
            'top_genres': genre_counts
        }
    
    def _calculate_audio_features(self, emotions):
        features = {
            'target_valence': 0.0,
            'target_energy': 0.0,
            'target_danceability': 0.0,
            'target_acousticness': 0.5,
            'target_tempo': 100.0
        }
        
        for emotion, weight in emotions.items():
            if emotion in self.mood_analyzer.emotion_mapping:
                for param, value in self.mood_analyzer.emotion_mapping[emotion].items():
                    features[param] += value * weight
        
        features['target_valence'] = float(np.clip(features['target_valence'], 0, 1))
        features['target_energy'] = float(np.clip(features['target_energy'], 0, 1))
        features['target_danceability'] = float(np.clip(features['target_danceability'], 0, 1))
        features['target_acousticness'] = float(features['target_acousticness'])
        features['target_tempo'] = float(features['target_tempo'])
        
        return features

    def _get_contextual_seeds(self, sp_client, dominant_emotion):
        if sp_client is None:
            raise Exception("Client Spotify non autenticato. Completa il flusso OAuth.")
        top_tracks = sp_client.current_user_top_tracks(limit=30)['items']
        if not top_tracks:
            print("Nessun top track trovato per l'utente.")
            return []
        
        
        track_ids = [track['id'] for track in top_tracks if track.get('id')]
        
        def batch(lst, n=50):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]
                
        features_list = []
        for batch_ids in batch(track_ids, 50):
            try:
                batch_features = sp_client.audio_features(batch_ids)
                if batch_features is not None:
                    features_list.extend(batch_features)
            except Exception as e:
                print(f"Errore nel recupero delle caratteristiche audio per il batch {batch_ids}: {e}")
                for tid in batch_ids:
                    try:
                        single_feature = sp_client.audio_features([tid])[0]
                        features_list.append(single_feature)
                    except Exception as e2:
                        print(f"Errore nel recupero delle caratteristiche audio per il track {tid}: {e2}")
        
        id_to_features = {feat['id']: feat for feat in features_list if feat and feat.get('id')}
        
        for track in top_tracks:
            tid = track.get('id')
            track['audio_features'] = id_to_features.get(tid)
        
        
        valid_tracks = [track for track in top_tracks if track.get('audio_features') is not None]
        
        target_features = self.mood_analyzer.emotion_mapping.get(
            dominant_emotion,
            self.mood_analyzer.emotion_mapping['joy']  # fallback
        )
        
        seeds = sorted(
            valid_tracks,
            key=lambda x: self._track_similarity(x, target_features),
            reverse=True
        )[:10]
        return seeds

    def _track_similarity(self, track, target_features):
        features = track.get('audio_features', {})
        similarity = 0.0
        if 'target_valence' in target_features:
            similarity += 1 - abs(features.get('valence', 0.5) - target_features['target_valence'])
        if 'target_energy' in target_features:
            similarity += 1 - abs(features.get('energy', 0.5) - target_features['target_energy'])
        return similarity
    
    def get_mood_recommendations(self, sp_client, user_input):
        if sp_client is None:
            raise Exception("Client Spotify non autenticato. Completa il flusso OAuth.")
            
        
        emotions = self.mood_analyzer.analyze_text(user_input)
        print(f"Emozioni rilevate: {emotions}")
        audio_features = self._calculate_audio_features(emotions)
        print(f"Caratteristiche audio: {audio_features}")
        dominant_emotion = max(emotions, key=emotions.get)
        
        mood_to_genres = {
            'joy': ['pop', 'dance', 'happy'],
            'sadness': ['sad', 'acoustic', 'piano'],
            'anger': ['rock', 'metal', 'intense'],
            'fear': ['ambient', 'instrumental'],
            'optimism': ['pop', 'indie', 'upbeat'],
            'surprise': ['electronic', 'experimental'],
            'love': ['pop', 'r-n-b', 'soul']
        }
        
        
        seed_genres = mood_to_genres.get(dominant_emotion.lower(), ['pop'])
        
        try:
            recs = sp_client.recommendations(
                seed_genres=seed_genres[:2], 
                limit=20,
                **audio_features
            )
            return recs.get('tracks', [])
        except Exception as e:
            print(f"Primo tentativo fallito: {e}")
            try:
                recs = sp_client.recommendations(
                    seed_genres=seed_genres[:2],
                    limit=20
                )
                return recs.get('tracks', [])
            except Exception as e2:
                print(f"Secondo tentativo fallito: {e2}")
                try:
                    recs = sp_client.recommendations(
                        seed_genres=['pop', 'rock'],
                        limit=20
                    )
                    return recs.get('tracks', [])
                except Exception as e3:
                    raise Exception(f"Impossibile ottenere raccomandazioni: {e3}")
    def _get_audio_features_with_retry(self, sp_client, track_ids, max_retries=3):
        """Ottieni le caratteristiche audio con retry in caso di errore"""
        for attempt in range(max_retries):
            try:
                return sp_client.audio_features(track_ids)
            except Exception as e:
                print(f"Tentativo {attempt+1} fallito: {e}")
                if attempt == max_retries - 1:
                    return []
                time.sleep(1)
    def create_mood_playlist(self, sp_client, playlist_name, track_ids):
        if sp_client is None:
            raise Exception("Client Spotify non autenticato. Completa il flusso OAuth.")
            
        
        user_id = sp_client.current_user()['id']
        
        playlist = sp_client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description="Playlist generata in base al tuo stato d'animo"
        )
        
        
        if track_ids:
            chunks = [track_ids[i:i+100] for i in range(0, len(track_ids), 100)]
            for chunk in chunks:
                sp_client.playlist_add_items(playlist['id'], chunk)
        return playlist['external_urls']['spotify']