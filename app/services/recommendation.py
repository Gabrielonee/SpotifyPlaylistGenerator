import random
import datetime
from app.models.emotion import Emotion
from app.utils.cache_manager import RecommendationCache

class RecommendationService:
    def get_available_genres(self, sp_client):
        """
        Restituisce la lista dei generi disponibili per le raccomandazioni da Spotify.
        """
        try:
            genres = sp_client.recommendation_genre_seeds()
            if isinstance(genres, dict) and 'genres' in genres:
                return genres['genres']
            return genres
        except Exception as e:
            print(f"Errore nel recupero dei generi disponibili: {e}")
            return ['pop']
    def __init__(self, spotify_service, mood_analysis_service):
        self.spotify_service = spotify_service
        self.mood_analysis_service = mood_analysis_service
        self.cache = RecommendationCache()
        self.familiar_proportion = 0.2
        self.last_cache_reset = datetime.datetime.now()
        self.cache_expiry_days = 7
        self.recommended_tracks_cache = set()
        self.mood_to_genres = {
            'joy': ['pop', 'dance', 'happy', 'disco', 'tropical', 'edm', 'funk', 'party'],
            'sadness': ['sad', 'acoustic', 'piano', 'indie', 'folk', 'ambient', 'chill', 'indie-pop'],
            'anger': ['rock', 'metal', 'intense', 'punk', 'hardcore', 'grunge', 'alt-rock', 'industrial'],
            'fear': ['ambient', 'instrumental', 'classical', 'cinematic', 'soundtracks', 'atmospheric'],
            'optimism': ['pop', 'indie', 'piano','upbeat', 'folk', 'gospel', 'soul', 'indie-pop', 'alt-rock'],
            'surprise': ['electronic', 'experimental', 'alternative', 'new-age', 'jazz', 'fusion', 'world-music'],
            'love': ['pop', 'r-n-b', 'soul', 'jazz', 'acoustic', 'singer-songwriter', 'indie', 'ballad']
        }
    
    def _calculate_audio_features(self, emotion):
        # emotion può essere un oggetto Emotion o un dict
        emotions_dict = emotion.emotions if hasattr(emotion, 'emotions') else emotion
        features = {
            'target_valence': 0.0,
            'target_energy': 0.0,
            'target_danceability': 0.0,
            'target_acousticness': 0.5,
            'target_tempo': 100.0
        }
        emotion_mapping = self.mood_analysis_service.get_emotion_mapping()
        for em, weight in emotions_dict.items():
            if em in emotion_mapping:
                for param, value in emotion_mapping[em].items():
                    features[param] += value * weight
        # Aggiunta di variazioni casuali
        for key in features:
            if key.startswith('target_'):
                if 'valence' in key or 'energy' in key or 'danceability' in key or 'acousticness' in key:
                    features[key] = max(0, min(1, features[key] + random.uniform(-0.1, 0.1)))
                elif 'tempo' in key:
                    features[key] = max(60, features[key] + random.uniform(-10, 10))
        additional_params = {
            'target_instrumentalness': random.uniform(0, 0.5),
            'target_liveness': random.uniform(0, 0.5),
            'min_popularity': random.randint(20, 40)
        }
        features.update(additional_params)
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
        
        target_features = self.mood_analysis_service.emotion_mapping.get(
            dominant_emotion,
            self.mood_analysis_service.emotion_mapping['joy']  # fallback
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
        
        # Non aggiungere tracce già consigliate, restituisci solo nuove se disponibili
        if new_tracks:
            return new_tracks
        else:
            print("Tutte le tracce sono già state consigliate, nessuna nuova disponibile. Prendo dalla Top 50 globale o playlist famose.")
            return self._get_global_popular_tracks()

    def _get_global_popular_tracks(self, limit=30):
        # Prova a prendere la Top 50 globale o playlist famose
        tracks = []
        try:
            # Top 50 Global playlist ufficiale Spotify
            top50 = self.spotify_service.sp.playlist("37i9dQZEVXbMDoHDwVN2tF")
            if 'tracks' in top50 and 'items' in top50['tracks']:
                for item in top50['tracks']['items']:
                    if item and 'track' in item and item['track']:
                        tracks.append(item['track'])
            # Se non bastano, aggiungi da altre playlist famose
            if len(tracks) < limit:
                playlist_ids = [
                    "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
                    "37i9dQZF1DX0XUsuxWHRQd",  # Hot Hits Italia
                    "37i9dQZF1DX4dyzvuaRJ0n",  # Top 50 Italia
                ]
                for pid in playlist_ids:
                    pl = self.spotify_service.sp.playlist(pid)
                    if 'tracks' in pl and 'items' in pl['tracks']:
                        for item in pl['tracks']['items']:
                            if item and 'track' in item and item['track']:
                                tracks.append(item['track'])
                            if len(tracks) >= limit:
                                break
                    if len(tracks) >= limit:
                        break
        except Exception as e:
            print(f"Errore nel recupero delle playlist globali: {e}")
        # Rimuovi duplicati
        unique = {}
        for t in tracks:
            if t.get('id') and t.get('id') not in unique:
                unique[t['id']] = t
        return list(unique.values())[:limit]

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
            
        emotions_dict = self.mood_analysis_service.analyze_text(user_input)
        print(f"Emozioni rilevate: {emotions_dict}")
        emotion = Emotion(emotions_dict)
        audio_features = self._calculate_audio_features(emotion.emotions)
        dominant_emotion = emotion.dominant_emotion
        mood_to_genres = {
            'joy': ['pop', 'dance', 'happy', 'disco', 'tropical', 'edm', 'funk', 'party'],
            'sadness': ['sad', 'acoustic', 'piano', 'indie', 'folk', 'ambient', 'chill', 'indie-pop'],
            'anger': ['rock', 'metal', 'intense', 'punk', 'hardcore', 'grunge', 'alt-rock', 'industrial'],
            'fear': ['ambient', 'instrumental', 'classical', 'cinematic', 'soundtracks', 'atmospheric'],
            'optimism': ['pop', 'indie', 'piano','upbeat', 'folk', 'gospel', 'soul', 'indie-pop', 'alt-rock'],
            'surprise': ['electronic', 'experimental', 'alternative', 'new-age', 'jazz', 'fusion', 'world-music'],
            'love': ['pop', 'r-n-b', 'soul', 'jazz', 'acoustic', 'singer-songwriter', 'indie', 'ballad']
        }
        
        familiar_tracks = self._get_familiar_tracks(sp_client)
        familiar_artist_ids = self._get_artists_from_tracks(familiar_tracks)
        
        target_new_tracks = int(30 * (1.0 - self.familiar_proportion))
        
        available_genres = self.get_available_genres(sp_client)
        
        # dominant_emotion è una stringa, mood_to_genres è un dict
        preferred_genres = mood_to_genres.get(str(dominant_emotion).lower(), ['pop'])
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
                datetime.time.sleep(1)
    
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