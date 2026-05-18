import random
import datetime
import logging

from app.models.emotion import Emotion
from app.utils.cache_manager import RecommendationCache

logger = logging.getLogger(__name__)

MOOD_TO_GENRES = {
    'joy':      ['pop', 'dance', 'happy', 'disco', 'tropical', 'edm', 'funk', 'party'],
    'sadness':  ['sad', 'acoustic', 'piano', 'indie', 'folk', 'ambient', 'chill', 'indie-pop'],
    'anger':    ['rock', 'metal', 'intense', 'punk', 'hardcore', 'grunge', 'alt-rock', 'industrial'],
    'fear':     ['ambient', 'instrumental', 'classical', 'cinematic', 'soundtracks', 'atmospheric'],
    'optimism': ['pop', 'indie', 'piano', 'upbeat', 'folk', 'gospel', 'soul', 'indie-pop', 'alt-rock'],
    'surprise': ['electronic', 'experimental', 'alternative', 'new-age', 'jazz', 'fusion', 'world-music'],
    'love':     ['pop', 'r-n-b', 'soul', 'jazz', 'acoustic', 'singer-songwriter', 'indie', 'ballad'],
}

MOOD_TO_SEARCH_TERMS = {
    'joy':      ['happy', 'joy', 'upbeat', 'dance', 'celebration', 'energetic', 'cheerful', 'ecstatic'],
    'sadness':  ['sad', 'melancholy', 'blue', 'nostalgia', 'heartbreak', 'sorrow', 'wistful', 'reflective'],
    'anger':    ['angry', 'intense', 'power', 'energy', 'furious', 'rage', 'aggressive', 'fierce'],
    'fear':     ['calm', 'relaxing', 'ambient', 'peaceful', 'soothing', 'serene', 'meditative', 'quiet'],
    'optimism': ['motivational', 'upbeat', 'inspiring', 'positive', 'hopeful', 'uplifting', 'bright', 'encouraging'],
    'surprise': ['discover', 'new', 'unusual', 'unexpected', 'exciting', 'different', 'unique', 'experimental'],
    'love':     ['love', 'romantic', 'passion', 'sweet', 'affection', 'tender', 'devotion', 'intimate'],
}

GLOBAL_FALLBACK_PLAYLISTS = [
    "37i9dQZEVXbMDoHDwVN2tF",  # Top 50 Global
    "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
    "37i9dQZF1DX0XUsuxWHRQd",  # Hot Hits Italia
    "37i9dQZF1DX4dyzvuaRJ0n",  # Top 50 Italia
]


class RecommendationService:
    def __init__(self, spotify_service, mood_analysis_service):
        self.spotify_service = spotify_service
        self.mood_analysis_service = mood_analysis_service
        self.cache = RecommendationCache()
        self.familiar_proportion = 0.2
        self.cache_expiry_days = 7

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_mood_recommendations(self, sp_client, emotion: Emotion):
        if sp_client is None:
            raise ValueError("Client Spotify non autenticato. Completa il flusso OAuth.")

        audio_features = self._calculate_audio_features(emotion)
        dominant_emotion = emotion.dominant_emotion
        logger.info("Emozione dominante: %s", dominant_emotion)

        familiar_tracks = self._get_familiar_tracks(sp_client)
        familiar_artist_ids = self._get_artists_from_tracks(familiar_tracks)
        target_new_count = int(30 * (1.0 - self.familiar_proportion))

        seed_genres = self._resolve_seed_genres(sp_client, dominant_emotion)
        new_recommendations = self._fetch_new_recommendations(
            sp_client, familiar_artist_ids, familiar_tracks, seed_genres, audio_features, target_new_count
        )

        filtered_new = self.cache.filter_tracks(new_recommendations)
        final = self._balance_recommendations(familiar_tracks, filtered_new)

        if final:
            return final

        if familiar_tracks:
            logger.warning("Fallback: restituisco solo tracce familiari")
            return familiar_tracks[:30]
        if new_recommendations:
            logger.warning("Fallback: restituisco solo nuove raccomandazioni")
            return new_recommendations[:30]

        logger.warning("Fallback: recupero da playlist pubbliche")
        fallback = self.get_fallback_tracks(sp_client, dominant_emotion)
        if fallback:
            return fallback

        raise RuntimeError("Impossibile ottenere raccomandazioni dopo molteplici tentativi")

    def create_mood_playlist(self, sp_client, playlist_name, track_ids):
        if sp_client is None:
            raise ValueError("Client Spotify non autenticato. Completa il flusso OAuth.")

        user_id = sp_client.current_user()['id']
        timestamp = datetime.datetime.now().strftime("%d-%m %H:%M")
        full_name = f"{playlist_name} [{timestamp}]"

        playlist = sp_client.user_playlist_create(
            user=user_id,
            name=full_name,
            public=False,
            description=f"Playlist generata in base al tuo stato d'animo il {timestamp}",
        )

        for chunk in [track_ids[i:i + 100] for i in range(0, len(track_ids), 100)]:
            sp_client.playlist_add_items(playlist['id'], chunk)

        return playlist['external_urls']['spotify']

    def get_fallback_tracks(self, sp_client, mood, limit=30):
        search_terms = MOOD_TO_SEARCH_TERMS.get(str(mood).lower(), ['popular', 'trending', 'hit'])
        if len(search_terms) > 3:
            search_terms = random.sample(search_terms, random.randint(2, 3))

        all_tracks = []
        for term in search_terms:
            all_tracks.extend(self._search_playlist_tracks(sp_client, term))
            if len(all_tracks) >= 20:
                break

        familiar = self._get_familiar_tracks(sp_client, 15)
        all_tracks.extend(familiar)

        unique = {t['id']: t for t in all_tracks if t.get('id')}
        return list(unique.values())[:limit]

    def get_available_genres(self, sp_client):
        try:
            genres = sp_client.recommendation_genre_seeds()
            return genres['genres'] if isinstance(genres, dict) and 'genres' in genres else genres
        except Exception:
            logger.exception("Errore nel recupero dei generi disponibili")
            return ['pop']

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_audio_features(self, emotion: Emotion):
        features = {
            'target_valence': 0.0,
            'target_energy': 0.0,
            'target_danceability': 0.0,
            'target_acousticness': 0.5,
            'target_tempo': 100.0,
        }
        emotion_mapping = self.mood_analysis_service.get_emotion_mapping()
        for em, weight in emotion.emotions.items():
            if em in emotion_mapping:
                for param, value in emotion_mapping[em].items():
                    features[param] += value * weight

        for key in features:
            if 'valence' in key or 'energy' in key or 'danceability' in key or 'acousticness' in key:
                features[key] = max(0.0, min(1.0, features[key] + random.uniform(-0.1, 0.1)))
            elif 'tempo' in key:
                features[key] = max(60.0, features[key] + random.uniform(-10, 10))

        features.update({
            'target_instrumentalness': random.uniform(0, 0.5),
            'target_liveness': random.uniform(0, 0.5),
            'min_popularity': random.randint(20, 40),
        })
        return features

    def _resolve_seed_genres(self, sp_client, dominant_emotion):
        available = self.get_available_genres(sp_client)
        preferred = MOOD_TO_GENRES.get(str(dominant_emotion).lower(), ['pop'])
        valid = [g for g in preferred if g in available]
        if len(valid) > 1:
            valid = random.sample(valid, random.randint(1, min(2, len(valid))))
        logger.debug("Generi seed: %s", valid)
        return valid

    def _fetch_new_recommendations(self, sp_client, artist_ids, familiar_tracks, seed_genres, audio_features, target):
        recommendations = []

        if artist_ids:
            try:
                seeds = random.sample(artist_ids, min(3, len(artist_ids)))
                recs = sp_client.recommendations(
                    seed_artists=seeds,
                    seed_genres=seed_genres[:1] if seed_genres else [],
                    limit=30,
                    **audio_features,
                )
                recommendations.extend(recs.get('tracks', []))
            except Exception:
                logger.warning("Strategia seed_artists fallita", exc_info=True)

        if len(recommendations) < target and familiar_tracks:
            try:
                seed_ids = random.sample([t['id'] for t in familiar_tracks if 'id' in t], min(2, len(familiar_tracks)))
                recs = sp_client.recommendations(
                    seed_tracks=seed_ids,
                    seed_genres=seed_genres[:1] if seed_genres else [],
                    limit=30,
                    **audio_features,
                )
                recommendations.extend(recs.get('tracks', []))
            except Exception:
                logger.warning("Strategia seed_tracks fallita", exc_info=True)

        if len(recommendations) < target and seed_genres:
            try:
                recs = sp_client.recommendations(
                    seed_genres=seed_genres[:3],
                    limit=30,
                    **audio_features,
                )
                recommendations.extend(recs.get('tracks', []))
            except Exception:
                logger.warning("Strategia seed_genres fallita", exc_info=True)

        unique = {t['id']: t for t in recommendations if t.get('id')}
        return list(unique.values())

    def _get_familiar_tracks(self, sp_client, limit=50):
        tracks = []
        for time_range in ('short_term', 'medium_term', 'long_term'):
            try:
                top = sp_client.current_user_top_tracks(time_range=time_range, limit=30)
                tracks.extend(top.get('items', []))
            except Exception:
                logger.warning("Errore top tracks (%s)", time_range, exc_info=True)

        try:
            recent = sp_client.current_user_recently_played(limit=30)
            tracks.extend(item['track'] for item in recent.get('items', []))
        except Exception:
            logger.warning("Errore recently played", exc_info=True)

        try:
            saved = sp_client.current_user_saved_tracks(limit=30)
            tracks.extend(item['track'] for item in saved.get('items', []))
        except Exception:
            logger.warning("Errore saved tracks", exc_info=True)

        unique = {t['id']: t for t in tracks if t.get('id')}
        result = list(unique.values())
        return random.sample(result, limit) if len(result) > limit else result

    def _get_artists_from_tracks(self, tracks):
        return list({
            artist['id']
            for track in tracks
            for artist in track.get('artists', [])
            if 'id' in artist
        })

    def _balance_recommendations(self, familiar_tracks, new_tracks, target_count=30):
        familiar_count = int(target_count * self.familiar_proportion)
        new_count = target_count - familiar_count

        result = []
        if familiar_tracks:
            pick = min(familiar_count, len(familiar_tracks))
            result.extend(random.sample(familiar_tracks, pick))
        if new_tracks:
            pick = min(new_count, len(new_tracks))
            result.extend(random.sample(new_tracks, pick))

        if len(result) < target_count:
            used = set(id(t) for t in result)
            extras = [t for t in (familiar_tracks + new_tracks) if id(t) not in used]
            needed = target_count - len(result)
            result.extend(random.sample(extras, min(needed, len(extras))))

        random.shuffle(result)
        return result[:target_count]

    def _get_global_popular_tracks(self, limit=30):
        tracks = []
        for pid in GLOBAL_FALLBACK_PLAYLISTS:
            try:
                pl = self.spotify_service.sp.playlist(pid)
                for item in pl.get('tracks', {}).get('items', []):
                    if item and item.get('track'):
                        tracks.append(item['track'])
                if len(tracks) >= limit:
                    break
            except Exception:
                logger.warning("Errore nel recupero playlist globale %s", pid, exc_info=True)

        unique = {t['id']: t for t in tracks if t.get('id')}
        return list(unique.values())[:limit]

    def _search_playlist_tracks(self, sp_client, term, max_playlists=10, tracks_per_playlist=15):
        tracks = []
        try:
            results = sp_client.search(q=term, type='playlist', limit=20)
            playlists = results.get('playlists', {}).get('items', [])
            if not playlists:
                return tracks
            selected = random.sample(playlists, min(max_playlists, len(playlists)))
            for pl in selected:
                try:
                    info = sp_client.playlist(pl['id'])
                    total = info.get('tracks', {}).get('total', 0)
                    offset = random.randint(0, max(0, min(total - tracks_per_playlist, 30))) if total > tracks_per_playlist else 0
                    pl_tracks = sp_client.playlist_tracks(pl['id'], limit=tracks_per_playlist, offset=offset)
                    for item in pl_tracks.get('items', []):
                        if item and item.get('track'):
                            tracks.append(item['track'])
                except Exception:
                    logger.debug("Errore nel recupero tracce dalla playlist %s", pl.get('id'), exc_info=True)
        except Exception:
            logger.warning("Errore ricerca playlist con termine '%s'", term, exc_info=True)
        return tracks
