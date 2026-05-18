import logging
from app.services.spotify_services import (
    get_spotify_client,
    get_auth_url as get_spotify_auth_url,
    process_callback as process_spotify_callback,
)

logger = logging.getLogger(__name__)


def get_auth_url(force=False):
    if force:
        return False, {'auth_url': get_spotify_auth_url()}

    sp_client = get_spotify_client()
    if sp_client is not None:
        user_profile = sp_client.current_user()
        return True, {'user_name': user_profile['display_name']}

    return False, {'auth_url': get_spotify_auth_url()}


def process_callback(code):
    try:
        result = process_spotify_callback(code)
        if result['success']:
            return {'success': True}
        return {'success': False, 'error': result.get('error', 'Unknown error')}
    except Exception as e:
        logger.exception("Errore durante il processo di callback")
        return {'success': False, 'error': str(e)}
