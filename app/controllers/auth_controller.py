from flask import session
from app.services.spotify_services import get_spotify_client, get_auth_url as get_spotify_auth_url, process_callback as process_spotify_callback

def get_auth_url(force=False):
    sp_client = get_spotify_client()
    if 'token_info' in session and not force:
        user_profile = sp_client.current_user()
        return True, {'user_name': user_profile['display_name']}
    else:
        auth_url = get_spotify_auth_url()
        return False, {'auth_url': auth_url}

def process_callback(code):
    try:
        result = process_spotify_callback(code)
        if result['success']:
            session['token_info'] = result['token_info']
            return {'success': True}
        else:
            return {'success': False, 'error': result.get('error', 'Unknown error')}
    except Exception as e:
        return {'success': False, 'error': str(e)}
