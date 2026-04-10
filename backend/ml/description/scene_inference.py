# backend/ml/description/scene_inference.py

from backend.ml.description.scene_scoring import SceneScoringEngine


class SceneInference:

    def __init__(self):
        self.scoring_engine = SceneScoringEngine()

    def infer(self, semantic_features, heuristic_features, model_predictions=None):
        return self.scoring_engine.score(
            semantic_features,
            heuristic_features,
            model_predictions=model_predictions,
        )
