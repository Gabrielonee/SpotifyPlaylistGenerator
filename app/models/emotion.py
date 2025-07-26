class Emotion:
    def get(self, key, default=None):
        return self.emotions.get(key, default)
    def __init__(self, emotions_dict=None):
        self.emotions = emotions_dict or {}
        
    @property
    def dominant_emotion(self):
        if not self.emotions:
            return 'joy'  # Default
        return max(self.emotions, key=self.emotions.get)

    def __iter__(self):
        return iter(self.emotions)

    def items(self):
        return self.emotions.items()
    
    def get_audio_features(self, emotion_mapping):
        features = {
            'target_valence': 0.0,
            'target_energy': 0.0,
            'target_danceability': 0.0,
            'target_acousticness': 0.5,
            'target_tempo': 100.0
        }
        
        for emotion, weight in self.emotions.items():
            if emotion in emotion_mapping:
                for param, value in emotion_mapping[emotion].items():
                    features[param] += value * weight
        
        return features