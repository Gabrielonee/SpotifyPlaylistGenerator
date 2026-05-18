import logging
from transformers import pipeline
from app.models.emotion import Emotion
from app.utils.translator import translate_to_english

logger = logging.getLogger(__name__)

EMOTION_MAPPING = {
    'joy':      {'target_valence': 0.85, 'target_energy': 0.75, 'target_danceability': 0.8},
    'sadness':  {'target_valence': 0.25, 'target_energy': 0.3,  'target_danceability': 0.3},
    'anger':    {'target_energy': 0.9,   'target_valence': 0.2,  'target_tempo': 140},
    'fear':     {'target_energy': 0.6,   'target_valence': 0.4,  'target_acousticness': 0.8},
    'surprise': {'target_energy': 0.7,   'target_valence': 0.6,  'target_loudness': -5},
    'love':     {'target_valence': 0.9,  'target_energy': 0.6,  'target_acousticness': 0.7},
    'optimism': {'target_valence': 0.8,  'target_energy': 0.7,  'target_danceability': 0.75},
}


class MoodAnalysisService:
    def __init__(self):
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=None,
        )
        self.emotion_mapping = EMOTION_MAPPING

    def analyze_text(self, text) -> Emotion:
        text = str(text)
        translated = translate_to_english(text)
        results = self.emotion_analyzer(translated)
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        total = sum(r['score'] for r in results)
        emotions_dict = {
            r['label'].lower(): r['score'] / total
            for r in results
        }
        logger.debug("Emozioni rilevate: %s", emotions_dict)
        return Emotion(emotions_dict)

    def get_emotion_mapping(self):
        return self.emotion_mapping
