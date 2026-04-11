# backend/ml/description/image_descriptor.py

from backend.ml.features.feature_extractor import FeatureExtractor
from backend.ml.description.advanced_semantic_extractor import (
    AdvancedSemanticFeatureExtractor,
)
from backend.ml.description.contextual_description_generator import (
    ContextualDescriptionGenerator,
)
from backend.ml.description.face_detection import FaceHeuristicDetector
from backend.ml.description.indoor_outdoor_detector import IndoorOutdoorDetector
from backend.ml.description.object_context import ObjectContextDetector
from backend.ml.description.scene_inference import SceneInference
from backend.ml.description.visual_heuristics import VisualHeuristicAnalyzer


class ImageDescriptor:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.semantic_extractor = AdvancedSemanticFeatureExtractor()
        self.heuristic_analyzer = VisualHeuristicAnalyzer()
        self.face_detector = FaceHeuristicDetector()
        self.object_detector = ObjectContextDetector()
        self.indoor_detector = IndoorOutdoorDetector()
        self.scene_inference = SceneInference()
        self.generator = ContextualDescriptionGenerator()

    def describe_from_reader(self, reader):
        # 1. features numéricas (reutilizas TODO tu pipeline)
        features = self.feature_extractor.extract_from_png_reader(reader)

        # 2. features semánticas
        semantic = self.semantic_extractor.extract(features)

        # 3. inferencia de escena
        heuristics = self.heuristic_analyzer.analyze(semantic, features)
        face = self.face_detector.detect(reader.pixels, reader.width, reader.height)
        object_context = self.object_detector.detect(
            features,
            features,
            features,
            features,
        )
        indoor = self.indoor_detector.detect(
            semantic,
            features,
            features,
            features,
        )
        scene = self.scene_inference.infer(
            semantic,
            heuristics,
            face_result=face,
            object_context=object_context,
            indoor_result=indoor,
        )

        # 4. generación de texto
        description = self.generator.generate(
            semantic,
            scene,
            heuristics,
            face,
            object_context,
            indoor,
        )

        return {
            "description": description,
            "semantic_features": semantic,
            "heuristic_features": heuristics,
            "face_detection": face,
            "object_context": object_context,
            "indoor_outdoor": indoor,
            "scene_inference": scene,
            "raw_features": features
        }
