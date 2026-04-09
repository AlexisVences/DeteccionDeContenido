# backend/ml/description/description_generator.py

class DescriptionGenerator:

    def generate(self, semantic_features, scene_tags):
        parts = []

        parts.append(f"Imagen {semantic_features['brightness']}")
        parts.append(semantic_features["contrast"])
        parts.append(f"estructura {semantic_features['complexity']}")
        parts.append(semantic_features["detail"])

        if scene_tags:
            parts.append("que parece ser " + ", ".join(scene_tags))

        return ", ".join(parts) + "."