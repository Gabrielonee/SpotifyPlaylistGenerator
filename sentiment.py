import time
import pandas as pd
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import random
import datetime

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
        
        # Cache per tracciare le tracce già consigliate
        self.recommended_tracks_cache = set()
        # Tempo di scadenza della cache in giorni
        self.cache_expiry_days = 7
        # Timestamp dell'ultimo reset della cache
        self.last_cache_reset = datetime.datetime.now()

    def translate_to_english(self, text):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception as e:
            print(f"Errore nella traduzione: {e}")
            return text

    def get_sentiment(self, prompt):
        #Translate to english
        prompt = self.translate_to_english(prompt)
        sentiment_score = self.analyze_sentiment(prompt)
        sentiment_label = self.label_sentiment(sentiment_score)
        return sentiment_label
    
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

    def get_available_genres(self, sp_client=None):
        if not sp_client:
            sp_client = self.sp
        try:
            return sp_client.recommendation_genre_seeds()['genres']
        except Exception as e:
            print(f"Errore nel recupero dei generi disponibili: {e}")
            return ['pop']  #Fallback to common genre
    
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
        
        
        #More diversity
        for key in features:
            if key.startswith('target_'):
                if 'valence' in key or 'energy' in key or 'danceability' in key or 'acousticness' in key:
                    features[key] = max(0, min(1, features[key] + random.uniform(-0.2, 0.2)))
                elif 'tempo' in key:
                    #More BPM diversity
                    features[key] = max(60, features[key] + random.uniform(-20, 20))
        
        additional_params = {
            'target_instrumentalness': random.uniform(0, 0.8),
            'target_liveness': random.uniform(0, 0.8),
            'min_popularity': random.randint(0, 40)  #Less popular tracks
        }
        
        features.update(additional_params)
        
        print(f"Caratteristiche audio con variabilità aumentata: {features}")
        
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
            self.mood_analyzer.emotion_mapping['joy']  #fallback
        )
        
        #We select 20 similar and 10 random
        top_20_similar = sorted(
            valid_tracks,
            key=lambda x: self._track_similarity(x, target_features),
            reverse=True
        )[:20]
        
        if len(top_20_similar) > 10:
            seeds = random.sample(top_20_similar, 10)
        else:
            seeds = top_20_similar
            
        return seeds

    def _track_similarity(self, track, target_features):
        features = track.get('audio_features', {})
        similarity = 0.0
        if 'target_valence' in target_features:
            similarity += 1 - abs(features.get('valence', 0.5) - target_features['target_valence'])
        if 'target_energy' in target_features:
            similarity += 1 - abs(features.get('energy', 0.5) - target_features['target_energy'])
        return similarity
        
    def _check_cache_expiry(self):
        
        now = datetime.datetime.now()
        days_passed = (now - self.last_cache_reset).days
        
        if days_passed >= self.cache_expiry_days:
            print(f"Reset della cache dopo {days_passed} giorni")
            self.recommended_tracks_cache = set()
            self.last_cache_reset = now
            return True
        return False

    def _filter_already_recommended(self, tracks):
        #Reset cache?
        self._check_cache_expiry()
        
        new_tracks = []
        for track in tracks:
            track_id = track.get('id')
            if track_id and track_id not in self.recommended_tracks_cache:
                new_tracks.append(track)
                #Add to cache
                self.recommended_tracks_cache.add(track_id)
        
        #If too filtered
        if len(new_tracks) < len(tracks) * 0.3 and len(tracks) > 0:
            print("Troppe tracce filtrate, aggiungendo alcune tracce già consigliate")
            already_recommended = [t for t in tracks if t.get('id') in self.recommended_tracks_cache]
            if already_recommended:
                #Max 30% tracks already recommended
                num_to_add = min(len(already_recommended), int(len(tracks) * 0.3))
                new_tracks.extend(random.sample(already_recommended, num_to_add))
        
        if new_tracks:
            return new_tracks
        else:
            print("Tutte le tracce sono già state consigliate, restituendo le tracce originali")
            return tracks

    def get_mood_recommendations(self, sp_client, user_input):
        if sp_client is None:
            raise Exception("Client Spotify non autenticato. Completa il flusso OAuth.")
            
        emotions = self.mood_analyzer.analyze_text(user_input)
        print(f"Emozioni rilevate: {emotions}")
        
        audio_features = self._calculate_audio_features(emotions)
        
        dominant_emotion = max(emotions, key=emotions.get)
        mood_to_genres = {
            'joy': ['pop', 'dance', 'happy', 'disco', 'tropical', 'edm', 'funk', 'party'],
            'sadness': ['sad', 'acoustic', 'piano', 'indie', 'folk', 'ambient', 'chill', 'indie-pop'],
            'anger': ['rock', 'metal', 'intense', 'punk', 'hardcore', 'grunge', 'alt-rock', 'industrial'],
            'fear': ['ambient', 'instrumental', 'classical', 'cinematic', 'soundtracks', 'atmospheric'],
            'optimism': ['pop', 'indie', 'upbeat', 'folk', 'gospel', 'soul', 'indie-pop', 'alt-rock'],
            'surprise': ['electronic', 'experimental', 'alternative', 'new-age', 'jazz', 'fusion', 'world-music'],
            'love': ['pop', 'r-n-b', 'soul', 'jazz', 'acoustic', 'singer-songwriter', 'indie', 'ballad']
        }
        
        available_genres = self.get_available_genres(sp_client)
        print(f"Generi disponibili: {available_genres}")
        
        preferred_genres = mood_to_genres.get(dominant_emotion.lower(), ['pop'])
        seed_genres = [genre for genre in preferred_genres if genre in available_genres]

        if len(seed_genres) > 1:
            seed_count = random.randint(1, min(3, len(seed_genres)))
            seed_genres = random.sample(seed_genres, seed_count)
        
        print(f"Usando i generi seed: {seed_genres}")
        recommendations = None
        
        strategies = ['genres', 'tracks', 'artists', 'search', 'fallback']
        random.shuffle(strategies)
        
        for strategy in strategies:
            try:
                if strategy == 'genres' and seed_genres:
                    print(f"Tentativo con seed_genres: {seed_genres}")
                    recs = sp_client.recommendations(
                        seed_genres=seed_genres, 
                        limit=30,  #More variety
                        **audio_features
                    )
                    recommendations = recs.get('tracks', [])
                    
                elif strategy == 'tracks':
                    print("Tentativo con seed_tracks")
                    time_ranges = ['short_term', 'medium_term', 'long_term']
                    random.shuffle(time_ranges)
                    selected_range = time_ranges[0]
                    
                    top_tracks = sp_client.current_user_top_tracks(time_range=selected_range, limit=50)
                    if top_tracks and 'items' in top_tracks and top_tracks['items']:
                        track_selection = random.sample(top_tracks['items'], min(4, len(top_tracks['items'])))
                        seed_tracks = [track['id'] for track in track_selection]
                        print(f"Usando seed_tracks da {selected_range}: {seed_tracks}")
                        recs = sp_client.recommendations(
                            seed_tracks=seed_tracks,
                            limit=30,
                            **audio_features
                        )
                        recommendations = recs.get('tracks', [])
                
                elif strategy == 'artists':
                    print("Tentativo con seed_artists")
                    time_ranges = ['short_term', 'medium_term', 'long_term']
                    random.shuffle(time_ranges)
                    selected_range = time_ranges[0]
                    
                    top_artists = sp_client.current_user_top_artists(time_range=selected_range, limit=50)
                    if top_artists and 'items' in top_artists and top_artists['items']:
                        artist_selection = random.sample(top_artists['items'], min(5, len(top_artists['items'])))
                        seed_artists = [artist['id'] for artist in artist_selection]
                        print(f"Usando seed_artists da {selected_range}: {seed_artists}")
                        recs = sp_client.recommendations(
                            seed_artists=seed_artists,
                            limit=40,
                            **audio_features
                        )
                        recommendations = recs.get('tracks', [])
                
                elif strategy == 'search':
                    print("Tentativo con ricerca di brani basati sull'emozione")
                    search_terms = mood_to_genres.get(dominant_emotion.lower(), ['pop'])
                    if len(search_terms) > 2:
                        search_terms = random.sample(search_terms, 2)
                    decades = ['60s', '70s', '80s', '90s', '2000s', '2010s', '2020s']
                    decade = random.choice(decades)
                    search_terms.append(decade)
                    
                    for term in search_terms:
                        track_results = sp_client.search(q=term, type='track', limit=35)
                        if track_results and 'tracks' in track_results and 'items' in track_results['tracks']:
                            recommendations = track_results['tracks']['items']
                            if recommendations:
                                break
                
                elif strategy == 'fallback':
                    print("Usando il metodo fallback per ottenere tracce da playlist pubbliche")
                    fallback_tracks = self.get_fallback_tracks(sp_client, dominant_emotion)
                    if fallback_tracks:
                        recommendations = fallback_tracks
                
                if recommendations:
                    print(f"Trovate {len(recommendations)} raccomandazioni con strategia {strategy}")
                    filtered_recommendations = self._filter_already_recommended(recommendations)
                    if filtered_recommendations:
                        random.shuffle(filtered_recommendations)
                        return filtered_recommendations
            
            except Exception as e:
                print(f"Tentativo con strategia {strategy} fallito: {e}")
                continue
        
        #Nothing worked
        raise Exception("Impossibile ottenere raccomandazioni dopo molteplici tentativi")
    
    def _get_audio_features_with_retry(self, sp_client, track_ids, max_retries=3):
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
        
        timestamp = datetime.datetime.now().strftime("%d-%m %H:%M")
        playlist_name = f"{playlist_name} [{timestamp}]"
        
        playlist = sp_client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description=f"Playlist generata in base al tuo stato d'animo il {timestamp}"
        )
        
        if track_ids:
            chunks = [track_ids[i:i+100] for i in range(0, len(track_ids), 100)]
            for chunk in chunks:
                sp_client.playlist_add_items(playlist['id'], chunk)
        return playlist['external_urls']['spotify']
    
    def get_fallback_tracks(self, sp_client, mood):
        mood_to_search = {
            'joy': ['happy', 'joy', 'festa', 'felicità', 'upbeat', 'dance', 'celebration', 'energetic', 'cheerful', 'ecstatic'],
            'sadness': ['sad', 'melancholy', 'tristezza', 'malinconia', 'blue', 'nostalgia', 'heartbreak', 'sorrow', 'wistful', 'reflective'],
            'anger': ['angry', 'intense', 'rabbia', 'intense', 'power', 'energy', 'furious', 'rage', 'aggressive', 'fierce'],
            'fear': ['calm', 'relaxing', 'rilassante', 'tranquillo', 'ambient', 'peaceful', 'soothing', 'serene', 'meditative', 'quiet'],
            'optimism': ['motivational', 'upbeat', 'motivazione', 'ottimismo', 'inspiring', 'positive', 'hopeful', 'uplifting', 'bright', 'encouraging'],
            'surprise': ['discover', 'new', 'scoperta', 'novità', 'unusual', 'unexpected', 'exciting', 'different', 'unique', 'experimental'],
            'love': ['love', 'romantic', 'amore', 'romantico', 'passion', 'sweet', 'affection', 'tender', 'devotion', 'intimate']
        }
        
        search_terms = mood_to_search.get(mood.lower(), ['popular', 'trending', 'hit'])
        
        
        if len(search_terms) > 3:
            search_terms = random.sample(search_terms, random.randint(2, 4))
        
        all_tracks = []
        
        for term in search_terms:
            try:
                print(f"Ricerca playlist con termine: {term}")
                playlist_results = sp_client.search(q=term, type='playlist', limit=50)  #Play with LIMIT
                
                if not playlist_results or 'playlists' not in playlist_results or 'items' not in playlist_results['playlists']:
                    print(f"Nessuna playlist trovata per il termine: {term}")
                    continue
                
                playlists = playlist_results['playlists']['items']
                if not playlists:
                    continue
            
                selected_playlists = random.sample(playlists, min(10, len(playlists)))
                
                for random_playlist in selected_playlists:
                    playlist_id = random_playlist['id']
                    
                    print(f"Usando playlist: {random_playlist['name']} (ID: {playlist_id})")
                    
                    playlist_info = sp_client.playlist(playlist_id)
                    if 'tracks' in playlist_info and 'total' in playlist_info['tracks']:
                        total_tracks = playlist_info['tracks']['total']
                        if total_tracks > 50:
                            offset = random.randint(0, min(total_tracks - 25, 75))
                        else:
                            offset = 0
                    else:
                        offset = 0
                    
                    playlist_tracks = sp_client.playlist_tracks(playlist_id, limit=30, offset=offset)
                    
                    if not playlist_tracks or 'items' not in playlist_tracks:
                        continue
                        
                    for item in playlist_tracks['items']:
                        if item and 'track' in item and item['track']:
                            all_tracks.append(item['track'])
                
                if len(all_tracks) >= 50:  
                    break
                    
            except Exception as e:
                print(f"Errore durante la ricerca di playlist per '{term}': {e}")
        
        if all_tracks:
            random.shuffle(all_tracks)
            filtered_tracks = self._filter_already_recommended(all_tracks)
            return filtered_tracks[:min(50, len(filtered_tracks))] 
        else:
            return []