class User:
    def __init__(self, user_profile=None):
        self.user_profile = user_profile or {}
        self.top_tracks = {}
        self.top_artists = {}
        self.recently_played = {}
        self.top_genres = {}
    
    @property
    def id(self):
        return self.user_profile.get('id')
    
    @property
    def display_name(self):
        return self.user_profile.get('display_name')
    
    def to_dict(self):
        return {
            'user_profile': self.user_profile,
            'top_tracks': self.top_tracks,
            'top_artists': self.top_artists,
            'recently_played': self.recently_played,
            'top_genres': self.top_genres
        }
    
    @classmethod
    def from_dict(cls, data):
        user = cls(data.get('user_profile'))
        user.top_tracks = data.get('top_tracks', {})
        user.top_artists = data.get('top_artists', {})
        user.recently_played = data.get('recently_played', {})
        user.top_genres = data.get('top_genres', {})
        return user