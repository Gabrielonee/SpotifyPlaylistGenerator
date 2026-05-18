import logging
from app.services.spotify_services import get_spotify_client, SpotifyService

logger = logging.getLogger(__name__)

_spotify_service = None
_mood_service = None
_rec_service = None


def _get_services():
    global _spotify_service, _mood_service, _rec_service
    if _spotify_service is None:
        from app.services.mood_analysis import MoodAnalysisService
        from app.services.recommendation import RecommendationService
        _spotify_service = SpotifyService()
        _mood_service = MoodAnalysisService()
        _rec_service = RecommendationService(_spotify_service, _mood_service)
    return _spotify_service, _mood_service, _rec_service


def get_user_recap_data():
    sp_client = get_spotify_client()
    if sp_client is None:
        return None

    spotify_service, _, _ = _get_services()
    user = spotify_service.get_user_data(sp_client)

    top_tracks = [
        {
            'name': t.get('name'),
            'artist': t.get('artists', [{}])[0].get('name'),
            'album': t.get('album', {}).get('name'),
            'image_url': t.get('album', {}).get('images', [{}])[0].get('url'),
            'url': t.get('external_urls', {}).get('spotify'),
        }
        for t in user.top_tracks.get('medium_term', {}).get('items', [])[:10]
    ]

    top_artists = [
        {
            'name': a.get('name'),
            'image_url': a.get('images', [{}])[0].get('url'),
            'url': a.get('external_urls', {}).get('spotify'),
        }
        for a in user.top_artists.get('medium_term', {}).get('items', [])[:10]
    ]

    recently_played = [
        {
            'name': item.get('track', {}).get('name'),
            'artist': item.get('track', {}).get('artists', [{}])[0].get('name'),
            'image': item.get('track', {}).get('album', {}).get('images', [{}])[0].get('url'),
            'url': item.get('track', {}).get('external_urls', {}).get('spotify'),
        }
        for item in user.recently_played.get('items', [])[:10]
    ]

    return {
        'user_name': user.display_name,
        'top_tracks': top_tracks,
        'top_artists': top_artists,
        'top_genres': list(user.top_genres.keys())[:10],
        'recently_played': recently_played,
    }


def process_recommendation_request(user_input):
    sp_client = get_spotify_client()
    if sp_client is None:
        return {'success': False, 'error': 'Utente non autenticato'}

    try:
        _, mood_service, rec_service = _get_services()
        emotion = mood_service.analyze_text(user_input)
        tracks = rec_service.get_mood_recommendations(sp_client, emotion)

        track_ids = [t['id'] for t in tracks if 'id' in t]
        playlist_url = None
        if track_ids:
            playlist_url = rec_service.create_mood_playlist(sp_client, "Playlist Mood", track_ids)

        return {
            'success': True,
            'data': {
                'analysis': emotion,
                'tracks': tracks,
                'user_input': user_input,
                'playlist_url': playlist_url,
            },
        }
    except Exception as e:
        logger.exception("Errore durante la generazione delle raccomandazioni")
        return {'success': False, 'error': str(e)}
