import datetime
import random

class RecommendationCache:
    def __init__(self, expiry_days=7):
        self.recommended_tracks = set()
        self.expiry_days = expiry_days
        self.last_reset = datetime.datetime.now()
    
    def check_expiry(self):
        now = datetime.datetime.now()
        days_passed = (now - self.last_reset).days
        
        if days_passed >= self.expiry_days:
            self.recommended_tracks = set()
            self.last_reset = now
            return True
        return False
    
    def add_track(self, track_id):
        self.recommended_tracks.add(track_id)
    
    def is_recommended(self, track_id):
        return track_id in self.recommended_tracks
    
    def filter_tracks(self, tracks, keep_ratio=0.2):
        self.check_expiry()
        
        new_tracks = []
        for track in tracks:
            track_id = track.get('id')
            if track_id and track_id not in self.recommended_tracks:
                new_tracks.append(track)
                self.add_track(track_id)
        
        if len(new_tracks) < len(tracks) * 0.5 and len(tracks) > 0:
            already_recommended = [t for t in tracks if t.get('id') in self.recommended_tracks]
            if already_recommended:
                num_to_add = min(len(already_recommended), int(len(tracks) * keep_ratio))
                new_tracks.extend(random.sample(already_recommended, num_to_add))
        
        return new_tracks or tracks