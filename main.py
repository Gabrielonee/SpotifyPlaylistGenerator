from flask import Flask, request, redirect, render_template, url_for
from sentiment import SpotifyMoodAnalyzer
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os

app = Flask(__name__)


client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI')

app.config['SESSION_COOKIE_NAME'] = 'mood_music_session'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600 

mood_analyzer = SpotifyMoodAnalyzer(client_id, client_secret, redirect_uri)


@app.route('/')
def home():
    sp_user = mood_analyzer.authenticate_user()
    
    if sp_user is None:
        auth_url = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=' '.join([
                'user-library-read',
                'user-top-read',
                'user-read-recently-played',
                'playlist-read-private',
                'playlist-modify-private',
                'playlist-modify-public'
            ]),
            cache_path=".spotify_cache",
            show_dialog=True
        ).get_authorize_url()
        
        return render_template('index.html', authenticated=False, login_url=auth_url)
    else:
        user_name = sp_user.current_user()['display_name']
        return render_template('index.html', authenticated=True, user_name=user_name)

@app.route('/login')
def login():
    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=' '.join([
            'user-library-read',
            'user-top-read',
            'user-read-recently-played',
            'playlist-read-private',
            'playlist-modify-private',
            'playlist-modify-public'
        ]),
        cache_path=".spotify_cache",
        show_dialog=True
    )
    
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return "Errore: codice di autorizzazione mancante. Per favore, <a href='/'>riprova</a>."
    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=' '.join([
            'user-library-read',
            'user-top-read',
            'user-read-recently-played',
            'playlist-read-private',
            'playlist-modify-private',
            'playlist-modify-public'
        ]),
        cache_path=".spotify_cache"
    )
    
    try:
        token_info = sp_oauth.get_access_token(code)
        sp_oauth._save_token_info(token_info)
        return redirect(url_for('home'))
    except Exception as e:
        return f"Errore durante l'autenticazione: {str(e)}. Per favore, <a href='/'>riprova</a>."
    
@app.route('/user_recap')
def user_recap():
    sp_user = mood_analyzer.authenticate_user()
    
    if sp_user is None:
        return redirect(url_for('login'))
    
    user_data = mood_analyzer.get_user_data(sp_user)
    return render_template('user_recap.html',
                           user=user_data['user_profile']['display_name'],
                           top_tracks=[{
                               'name': track['name'],
                               'artist': track['artists'][0]['name'],
                               'album': track['album']['name'],
                               'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                               'url': track['external_urls']['spotify']
                           } for track in user_data.get('top_tracks', {}).get('short_term', {}).get('items', [])],
                           top_artists=[artist['name'] for artist in user_data.get('top_artists', {}).get('short_term', {}).get('items', [])[:5]],
                           top_genres=list(user_data.get('top_genres', {}).keys())[:5],
                           recently_played=[{
                               'name': track['track']['name'],
                               'artist': track['track']['artists'][0]['name'],
                               'image': track['track']['album']['images'][0]['url'],
                               'url': track['track']['external_urls']['spotify']
                           } for track in user_data.get('recently_played', {}).get('items', [])[:5]]
                           )

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form['user_input']
        print(f"Input utente: {user_input}")  # Debug
        
        sp_user = mood_analyzer.authenticate_user()
        if sp_user is None:
            return redirect(url_for('login'))
        user_input = request.form['user_input']
        analysis = mood_analyzer.analyze_text(user_input)
        print(f"Analisi del testo: {analysis}")  # Debug
        
        recommendations = mood_analyzer.get_mood_recommendations(sp_user, user_input)
        print(f"Numero di raccomandazioni ottenute: {len(recommendations)}")  # Debug
        
        if not recommendations:
            return render_template('error.html', 
                                  error="Non è stato possibile generare raccomandazioni per questo input. Prova con un altro testo.")
        
        track_ids = [track['id'] for track in recommendations if 'id' in track]
        if not track_ids:
            return render_template('error.html', 
                                  error="Non è stato possibile estrarre ID delle tracce validi dalle raccomandazioni.")
        
        playlist_url = None
        if track_ids:
            playlist_url = mood_analyzer.create_mood_playlist(
                sp_user,
                f"Playlist per: {user_input[:30]}",
                track_ids
            )
        
        return render_template('recommendations.html',
                             user_input=user_input,
                             analysis=analysis,
                             tracks=recommendations,
                             playlist_url=playlist_url)
    
    except Exception as e:
        print(f"Errore: {str(e)}")  # Debug
        return render_template('error.html', error=str(e))
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)