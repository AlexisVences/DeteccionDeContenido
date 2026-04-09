# backend/ml/description/image_descriptor.py

from backend.ml.features.feature_extractor import FeatureExtractor
from backend.ml.description.semantic_extractor import SemanticFeatureExtractor
from backend.ml.description.scene_inference import SceneInference
from backend.ml.description.description_generator import DescriptionGenerator


class ImageDescriptor:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor()
        self.scene_inference = SceneInference()
        self.generator = DescriptionGenerator()

    def describe_from_reader(self, reader):
        # 1. features numéricas (reutilizas TODO tu pipeline)
        features = self.feature_extractor.extract_from_png_reader(reader)

        # 2. features semánticas
        semantic = self.semantic_extractor.extract(features)

        # 3. inferencia de escena
        scene = self.scene_inference.infer(semantic)

        # 4. generación de texto
        description = self.generator.generate(semantic, scene)

        return {
            "description": description,
            "semantic_features": semantic,
            "raw_features": features
        }