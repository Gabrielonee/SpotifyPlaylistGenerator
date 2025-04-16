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
        
        self.familiar_proportion = 0.6 
        self.similarity_threshold = 0.7  #Similarity thresold
    
    #Translate to english
    def translate_to_english(self, text):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception as e:
            print(f"Errore nella traduzione: {e}")
            return text

    def get_sentiment(self, prompt):
        
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
            'short_term': sp_user.current_user_top_tracks(time_range='short_term', limit=20),
            'medium_term': sp_user.current_user_top_tracks(time_range='medium_term', limit=30),
            'long_term': sp_user.current_user_top_tracks(time_range='long_term', limit=40)
        }
    
        top_artists = {
            'short_term': sp_user.current_user_top_artists(time_range='short_term', limit=20),
            'medium_term': sp_user.current_user_top_artists(time_range='medium_term', limit=30),
            'long_term': sp_user.current_user_top_artists(time_range='long_term', limit=40)
        }
        
        recently_played = sp_user.current_user_recently_played(limit=30)
        
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
        
        
        #More similar songs to the userrr
        for key in features:
            if key.startswith('target_'):
                if 'valence' in key or 'energy' in key or 'danceability' in key or 'acousticness' in key:
                    features[key] = max(0, min(1, features[key] + random.uniform(-0.1, 0.1)))
                elif 'tempo' in key:
                    #Less BPM diversity 
                    features[key] = max(60, features[key] + random.uniform(-10, 10))
        
        additional_params = {
            'target_instrumentalness': random.uniform(0, 0.5),
            'target_liveness': random.uniform(0, 0.5),
            'min_popularity': random.randint(20, 40)  #More familiar tracks
        }
        
        features.update(additional_params)
        
        print(f"Caratteristiche audio calcolate: {features}")
        
        return features

    def _get_contextual_seeds(self, sp_client, dominant_emotion):
        if sp_client is None:
            raise Exception("Client Spotify non autenticato. Completa il flusso OAuth.")
        
        #Recent and top tracks combined
        top_tracks = sp_client.current_user_top_tracks(limit=50)['items']
        recent_tracks = sp_client.current_user_recently_played(limit=30)
        recent_track_items = [item['track'] for item in recent_tracks['items']] if 'items' in recent_tracks else []
        
        combined_tracks = top_tracks + recent_track_items
        if not combined_tracks:
            print("Nessuna traccia trovata per l'utente.")
            return []
        
        unique_tracks = {}
        for track in combined_tracks:
            if track.get('id') and track.get('id') not in unique_tracks:
                unique_tracks[track.get('id')] = track
        
        combined_tracks = list(unique_tracks.values())
        
        track_ids = [track['id'] for track in combined_tracks if track.get('id')]
        
        def batch(lst, n=50):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]
                
        features_list = []
        for batch_ids in batch(track_ids, 50):
            try:
                batch_features = sp_client.audio_features(batch_ids)
                if batch_features is not None:
                    features_list.extend([f for f in batch_features if f])
            except Exception as e:
                print(f"Errore nel recupero delle caratteristiche audio per il batch: {e}")
                for tid in batch_ids:
                    try:
                        single_feature = sp_client.audio_features([tid])[0]
                        if single_feature:
                            features_list.append(single_feature)
                    except Exception as e2:
                        print(f"Errore nel recupero delle caratteristiche audio per il track {tid}: {e2}")
        
        id_to_features = {feat['id']: feat for feat in features_list if feat and feat.get('id')}
        
        for track in combined_tracks:
            tid = track.get('id')
            if tid in id_to_features:
                track['audio_features'] = id_to_features[tid]
        
        valid_tracks = [track for track in combined_tracks if track.get('audio_features') is not None]
        
        target_features = self.mood_analyzer.emotion_mapping.get(
            dominant_emotion,
            self.mood_analyzer.emotion_mapping['joy']  # fallback
        )
        
        #More similar track (30)
        top_similar = sorted(
            valid_tracks,
            key=lambda x: self._track_similarity(x, target_features),
            reverse=True
        )[:30]
        
        if len(top_similar) > 10:
            seeds = random.sample(top_similar, 10)
        else:
            seeds = top_similar
            
        return seeds

    def _track_similarity(self, track, target_features):
        features = track.get('audio_features', {})
        similarity = 0.0
        
        if 'target_valence' in target_features:
            similarity += 1 - abs(features.get('valence', 0.5) - target_features['target_valence'])
        if 'target_energy' in target_features:
            similarity += 1 - abs(features.get('energy', 0.5) - target_features['target_energy'])
        if 'target_danceability' in target_features:
            similarity += 1 - abs(features.get('danceability', 0.5) - target_features['target_danceability'])
        if 'target_acousticness' in target_features:
            similarity += 1 - abs(features.get('acousticness', 0.5) - target_features['target_acousticness'])
        if 'target_tempo' in target_features:
            tempo_diff = abs(features.get('tempo', 120) - target_features['target_tempo']) / 100
            similarity += 1 - min(1, tempo_diff)
            
        num_features = sum(1 for f in target_features if f.startswith('target_'))
        if num_features > 0:
            similarity = similarity / num_features
            
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
        # Reset cache?
        self._check_cache_expiry()
        
        new_tracks = []
        for track in tracks:
            track_id = track.get('id')
            if track_id and track_id not in self.recommended_tracks_cache:
                new_tracks.append(track)
                # Add to cache
                self.recommended_tracks_cache.add(track_id)
        
        if len(new_tracks) < len(tracks) * 0.5 and len(tracks) > 0:
            print("Tracce filtrate, aggiungendo alcune tracce già consigliate in proporzione ridotta")
            already_recommended = [t for t in tracks if t.get('id') in self.recommended_tracks_cache]
            if already_recommended:
                # Max 20% tracks already recommended (before qwas 30%)
                num_to_add = min(len(already_recommended), int(len(tracks) * 0.2))
                new_tracks.extend(random.sample(already_recommended, num_to_add))
        
        if new_tracks:
            return new_tracks
        else:
            print("Tutte le tracce sono già state consigliate, restituendo le tracce originali")
            return tracks

    def _get_familiar_tracks(self, sp_client, limit=50):
        familiar_tracks = []
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                top = sp_client.current_user_top_tracks(time_range=time_range, limit=30)
                if top and 'items' in top:
                    familiar_tracks.extend(top['items'])
            except Exception as e:
                print(f"Errore nel recupero delle top tracks ({time_range}): {e}")
        
        #recent tracks
        try:
            recent = sp_client.current_user_recently_played(limit=30)
            if recent and 'items' in recent:
                familiar_tracks.extend([item['track'] for item in recent['items']])
        except Exception as e:
            print(f"Errore nel recupero delle tracce recenti: {e}")
        
        #saved tracks
        try:
            saved = sp_client.current_user_saved_tracks(limit=30)
            if saved and 'items' in saved:
                familiar_tracks.extend([item['track'] for item in saved['items']])
        except Exception as e:
            print(f"Errore nel recupero delle tracce salvate: {e}")
            
        unique_tracks = {}
        for track in familiar_tracks:
            if track.get('id') and track.get('id') not in unique_tracks:
                unique_tracks[track.get('id')] = track
                
        result = list(unique_tracks.values())
        if len(result) > limit:
            return random.sample(result, limit)
        return result

    def _get_artists_from_tracks(self, tracks):
        artist_ids = set()
        for track in tracks:
            if 'artists' in track:
                for artist in track['artists']:
                    if 'id' in artist:
                        artist_ids.add(artist['id'])
        return list(artist_ids)

    def _balance_recommendations(self, familiar_tracks, new_tracks, target_count=30):
        familiar_count = int(target_count * self.familiar_proportion)
        new_count = target_count - familiar_count
        
        print(f"Bilanciamento: {familiar_count} familiari, {new_count} nuove")
        
        result = []
        
        #Adding familiar tracks
        if familiar_tracks:
            if len(familiar_tracks) > familiar_count:
                result.extend(random.sample(familiar_tracks, familiar_count))
            else:
                result.extend(familiar_tracks)
                
        #Adding new tracks
        if new_tracks:
            if len(new_tracks) > new_count:
                result.extend(random.sample(new_tracks, new_count))
            else:
                result.extend(new_tracks)
                
        remaining = target_count - len(result)
        if remaining > 0:
            extra_familiar = []
            extra_new = []
            
            if len(familiar_tracks) > len(result):
                remaining_familiar = [t for t in familiar_tracks if t not in result]
                extra_familiar = random.sample(remaining_familiar, min(remaining, len(remaining_familiar)))
                
            if len(extra_familiar) < remaining and len(new_tracks) > 0:
                remaining_new = [t for t in new_tracks if t not in result]
                extra_new = random.sample(remaining_new, min(remaining - len(extra_familiar), len(remaining_new)))
                
            result.extend(extra_familiar + extra_new)
            
        random.shuffle(result)
        return result[:target_count]

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
        
        familiar_tracks = self._get_familiar_tracks(sp_client)
        familiar_artist_ids = self._get_artists_from_tracks(familiar_tracks)
        

        target_new_tracks = int(30 * (1.0 - self.familiar_proportion))
        
        available_genres = self.get_available_genres(sp_client)
        
        preferred_genres = mood_to_genres.get(dominant_emotion.lower(), ['pop'])
        seed_genres = [genre for genre in preferred_genres if genre in available_genres]

        if len(seed_genres) > 1:
            seed_count = random.randint(1, min(2, len(seed_genres)))
            seed_genres = random.sample(seed_genres, seed_count)
        
        print(f"Usando i generi seed: {seed_genres}")
        new_recommendations = []
        
        if familiar_artist_ids:
            try:
                seed_artists = random.sample(familiar_artist_ids, min(3, len(familiar_artist_ids)))
                print(f"Usando artisti familiari come seed: {seed_artists}")
                
                recs = sp_client.recommendations(
                    seed_artists=seed_artists,
                    seed_genres=seed_genres[:1] if seed_genres else [],
                    limit=30,
                    **audio_features
                )
                
                if recs and 'tracks' in recs:
                    new_recommendations.extend(recs['tracks'])
            except Exception as e:
                print(f"Errore usando seed_artists: {e}")
        
        #Second strategy
        if len(new_recommendations) < target_new_tracks and familiar_tracks:
            try:
                seed_tracks = random.sample([t['id'] for t in familiar_tracks if 'id' in t], 
                                           min(2, len(familiar_tracks)))
                
                print(f"Usando tracce familiari come seed: {seed_tracks}")
                
                recs = sp_client.recommendations(
                    seed_tracks=seed_tracks,
                    seed_genres=seed_genres[:1] if seed_genres else [],
                    limit=30,
                    **audio_features
                )
                
                if recs and 'tracks' in recs:
                    new_recommendations.extend(recs['tracks'])
            except Exception as e:
                print(f"Errore usando seed_tracks: {e}")
        
        # 3 strategy
        if len(new_recommendations) < target_new_tracks and seed_genres:
            try:
                print(f"Usando solo generi come seed: {seed_genres}")
                
                recs = sp_client.recommendations(
                    seed_genres=seed_genres[:3],
                    limit=30,
                    **audio_features
                )
                
                if recs and 'tracks' in recs:
                    new_recommendations.extend(recs['tracks'])
            except Exception as e:
                print(f"Errore usando seed_genres: {e}")
        
        unique_new_tracks = {}
        for track in new_recommendations:
            if track.get('id') and track.get('id') not in unique_new_tracks:
                unique_new_tracks[track.get('id')] = track
        
        new_recommendations = list(unique_new_tracks.values())
        
        filtered_new = self._filter_already_recommended(new_recommendations)
        final_recommendations = self._balance_recommendations(familiar_tracks, filtered_new)
        
        if final_recommendations:
            return final_recommendations
        

        if familiar_tracks:
            print("Usando solo tracce familiari come fallback")
            return familiar_tracks[:min(30, len(familiar_tracks))]
        elif new_recommendations:
            print("Usando solo nuove raccomandazioni come fallback")
            return new_recommendations[:min(30, len(new_recommendations))]
        else:
            print("Usando il metodo fallback per ottenere tracce da playlist pubbliche")
            fallback_tracks = self.get_fallback_tracks(sp_client, dominant_emotion)
            if fallback_tracks:
                return fallback_tracks
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
            search_terms = random.sample(search_terms, random.randint(2, 3))
        
        all_tracks = []
        
        
        familiar_tracks = self._get_familiar_tracks(sp_client, 15)
        
        for term in search_terms:
            try:
                print(f"Ricerca playlist con termine: {term}")
                playlist_results = sp_client.search(q=term, type='playlist', limit=20)  #50 --> 20
                
                if not playlist_results or 'playlists' not in playlist_results or 'items' not in playlist_results['playlists']:
                    print(f"Nessuna playlist trovata per il termine: {term}")
                    continue
                
                playlists = playlist_results['playlists']['items']
                if not playlists:
                    continue
            
                #Less playlist to see
                selected_playlists = random.sample(playlists, min(10, len(playlists)))
                
                for random_playlist in selected_playlists:
                    playlist_id = random_playlist['id']
                    
                    print(f"Usando playlist: {random_playlist['name']} (ID: {playlist_id})")
                    
                    playlist_info = sp_client.playlist(playlist_id)
                    if 'tracks' in playlist_info and 'total' in playlist_info['tracks']:
                        total_tracks = playlist_info['tracks']['total']
                        if total_tracks > 30:
                            offset = random.randint(0, min(total_tracks - 15, 30))
                        else:
                            offset = 0
                    else:
                        offset = 0
                    
                    playlist_tracks = sp_client.playlist_tracks(playlist_id, limit=15, offset=offset)
                    
                    if not playlist_tracks or 'items' not in playlist_tracks:
                        continue
                        
                    for item in playlist_tracks['items']:
                        if item and 'track' in item and item['track']:
                            all_tracks.append(item['track'])
                
                if len(all_tracks) >= 20:
                    break
            except Exception as e:
                print(f"Errore durante la ricerca con il termine '{term}': {e}")
        
        if not all_tracks:
            print("Nessuna traccia trovata tramite il metodo fallback, utilizzando tracce familiari.")
            return familiar_tracks
        
        unique_fallback = {}
        for track in all_tracks:
            if track.get('id') and track.get('id') not in unique_fallback:
                unique_fallback[track.get('id')] = track
        fallback_list = list(unique_fallback.values())
        
        if len(fallback_list) < 20:
            fallback_list.extend(familiar_tracks)
            unique_fallback = {}
            for track in fallback_list:
                if track.get('id') and track.get('id') not in unique_fallback:
                    unique_fallback[track.get('id')] = track
            fallback_list = list(unique_fallback.values())
        return fallback_list[:30]