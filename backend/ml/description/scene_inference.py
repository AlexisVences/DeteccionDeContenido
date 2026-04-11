# backend/ml/description/scene_inference.py

from backend.ml.description.advanced_scene_scoring import AdvancedSceneScoringEngine


class SceneInference:

    def __init__(self):
        self.scoring_engine = AdvancedSceneScoringEngine()

    def infer(
        self,
        semantic_features,
        heuristic_features,
        face_result=None,
        object_context=None,
        indoor_result=None,
        model_predictions=None,
    ):
        return self.scoring_engine.score(
            semantic_features,
            heuristic_features,
            face_result=face_result,
            object_context=object_context,
            indoor_result=indoor_result,
            model_predictions=model_predictions,
        )
