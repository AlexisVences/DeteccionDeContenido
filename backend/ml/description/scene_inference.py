# backend/ml/description/scene_inference.py

class SceneInference:

    def infer(self, semantic_features):
        scene = []

        if semantic_features["brightness"] == "brillante":
            scene.append("posiblemente exterior")

        if semantic_features["texture"] == "uniforme":
            scene.append("escena simple")

        if semantic_features["detail"] == "rica en detalles":
            scene.append("escena compleja")

        return scene