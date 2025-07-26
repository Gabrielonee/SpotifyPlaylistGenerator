from transformers import pipeline
from app.models.emotion import Emotion
from app.utils.translator import translate_to_english

class MoodAnalysisService:
    def __init__(self):
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=None
        )
        
        self.emotion_mapping = {
            'joy': {'target_valence': 0.85, 'target_energy': 0.75, 'target_danceability': 0.8},
            'sadness': {'target_valence': 0.25, 'target_energy': 0.3, 'target_danceability': 0.3},
            'anger': {'target_energy': 0.9, 'target_valence': 0.2, 'target_tempo': 140},
            'fear': {'target_energy': 0.6, 'target_valence': 0.4, 'target_acousticness': 0.8},
            'surprise': {'target_energy': 0.7, 'target_valence': 0.6, 'target_loudness': -5},
            'love': {'target_valence': 0.9, 'target_energy': 0.6, 'target_acousticness': 0.7},
            'optimism': {'target_valence': 0.8, 'target_energy': 0.7, 'target_danceability': 0.75}
        }
    
    def analyze_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        # Traduzione in inglese per migliori risultati
        translated_text = translate_to_english(text)
        results = self.emotion_analyzer(translated_text)
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        total = sum(res['score'] for res in results)
        emotions_dict = {
            str(res['label']).lower(): res['score'] / total
            for res in results
        }
        
        return Emotion(emotions_dict)
    
    def get_emotion_mapping(self):
        return self.emotion_mapping