from flask import Blueprint, render_template, redirect, url_for, request
from app.controllers.auth_controller import get_auth_url, process_callback
from app.controllers.music_controller import get_user_recap_data, process_recommendation_request

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    authenticated, data = get_auth_url()
    if authenticated:
        return render_template('index.html', authenticated=True, user_name=data['user_name'])
    else:
        return render_template('index.html', authenticated=False, login_url=data['auth_url'])

@main_bp.route('/login')
def login():
    _, data = get_auth_url(force=True)
    return redirect(data['auth_url'])

@main_bp.route('/callback')
def callback():
    result = process_callback(request.args.get('code'))
    if result['success']:
        return redirect(url_for('main.home'))
    else:
        return f"Errore durante l'autenticazione: {result['error']}. Per favore, <a href='/'>riprova</a>."

@main_bp.route('/user_recap')
def user_recap():
    user_data = get_user_recap_data()
    if user_data is None:
        return redirect(url_for('main.login'))
    return render_template('user_recap.html', **user_data)

@main_bp.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form['user_input']
        result = process_recommendation_request(user_input)
        if result['success']:
            # Passa sempre analysis, tracks, user_input, playlist_url
            return render_template('recommendations.html', **result['data'])
        else:
            return render_template('error.html', error=result['error'])
    except Exception as e:
        return render_template('error.html', error=str(e))