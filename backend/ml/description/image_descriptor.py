# backend/ml/description/image_descriptor.py

from backend.ml.features.feature_extractor import FeatureExtractor
from backend.ml.description.advanced_semantic_extractor import (
    AdvancedSemanticFeatureExtractor,
)
from backend.ml.description.scene_inference import SceneInference
from backend.ml.description.description_generator import DescriptionGenerator
from backend.ml.description.visual_heuristics import VisualHeuristicAnalyzer


class ImageDescriptor:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.semantic_extractor = AdvancedSemanticFeatureExtractor()
        self.heuristic_analyzer = VisualHeuristicAnalyzer()
        self.scene_inference = SceneInference()
        self.generator = DescriptionGenerator()

    def describe_from_reader(self, reader):
        # 1. features numéricas (reutilizas TODO tu pipeline)
        features = self.feature_extractor.extract_from_png_reader(reader)

        # 2. features semánticas
        semantic = self.semantic_extractor.extract(features)

        # 3. inferencia de escena
        heuristics = self.heuristic_analyzer.analyze(semantic, features)
        scene = self.scene_inference.infer(semantic, heuristics)

        # 4. generación de texto
        description = self.generator.generate(semantic, scene, heuristics)

        return {
            "description": description,
            "semantic_features": semantic,
            "heuristic_features": heuristics,
            "scene_inference": scene,
            "raw_features": features
        }
