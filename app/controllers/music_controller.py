
from app.services.spotify_services import get_spotify_client, SpotifyService
from app.services.mood_analysis import MoodAnalysisService
from app.services.recommendation import RecommendationService

spotify_service = SpotifyService()
mood_analysis_service = MoodAnalysisService()
rec_service = RecommendationService(spotify_service, mood_analysis_service)

def get_user_recap_data():
    sp_client = get_spotify_client()
    if not sp_client:
        return None
    user = spotify_service.get_user_data(sp_client)
    # Top tracks
    top_tracks = []
    for track in user.top_tracks.get('medium_term', {}).get('items', [])[:10]:
        top_tracks.append({
            'name': track.get('name'),
            'artist': track.get('artists', [{}])[0].get('name'),
            'album': track.get('album', {}).get('name'),
            'image_url': track.get('album', {}).get('images', [{}])[0].get('url'),
            'url': track.get('external_urls', {}).get('spotify')
        })
    # Top artists
    top_artists = []
    for artist in user.top_artists.get('medium_term', {}).get('items', [])[:10]:
        top_artists.append({
            'name': artist.get('name'),
            'image_url': artist.get('images', [{}])[0].get('url'),
            'url': artist.get('external_urls', {}).get('spotify')
        })
    # Top genres
    top_genres = list(user.top_genres.keys())[:10]
    # Recently played
    recently_played = []
    for item in user.recently_played.get('items', [])[:10]:
        track = item.get('track', {})
        recently_played.append({
            'name': track.get('name'),
            'artist': track.get('artists', [{}])[0].get('name'),
            'image': track.get('album', {}).get('images', [{}])[0].get('url'),
            'url': track.get('external_urls', {}).get('spotify')
        })
    return {
        'user_name': user.display_name,
        'top_tracks': top_tracks,
        'top_artists': top_artists,
        'top_genres': top_genres,
        'recently_played': recently_played
    }

def process_recommendation_request(user_input):
    sp_client = get_spotify_client()
    if not sp_client:
        return {'success': False, 'error': 'Utente non autenticato'}
    try:
        # Analisi emozioni
        emotions_dict = mood_analysis_service.analyze_text(user_input)
        # Raccomandazioni
        recommendations = rec_service.get_mood_recommendations(sp_client, user_input)
        # Crea la playlist su Spotify e ottieni il link reale
        track_ids = [t['id'] for t in recommendations if 'id' in t]
        playlist_url = None
        if track_ids:
            playlist_url = rec_service.create_mood_playlist(sp_client, "Playlist Mood", track_ids)
        return {
            'success': True,
            'data': {
                'analysis': emotions_dict,
                'tracks': recommendations,
                'user_input': user_input,
                'playlist_url': playlist_url
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
