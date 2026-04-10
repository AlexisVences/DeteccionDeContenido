# backend/ml/description/description_generator.py


class DescriptionGenerator:

    def generate(self, semantic_features, scene_result, heuristic_features):
        scene_label = self._scene_label(scene_result["top_scene"])
        palette_hint = semantic_features["palette_hint"]
        spatial_layout = semantic_features["spatial_layout"]
        texture = semantic_features["texture"]
        edge_profile = semantic_features["edge_profile"]
        confidence = scene_result["confidence"]

        description_parts = [
            scene_label,
            f"con {palette_hint}",
        ]

        if spatial_layout != "distribucion sin patron fuerte":
            description_parts.append(spatial_layout)

        description_parts.append(
            f"textura {texture} y {edge_profile}"
        )

        evidence = scene_result["evidence"][:2]
        if evidence:
            description_parts.append(
                "lo que sugiere " + self._suggestion_phrase(scene_result["top_scene"], confidence)
            )

        sentence = ", ".join(description_parts) + "."

        if heuristic_features["evidence"]:
            evidence_sentence = "Senales observadas: " + ", ".join(
                heuristic_features["evidence"][:4]
            ) + "."
            return sentence + " " + evidence_sentence

        return sentence

    def _scene_label(self, scene_name):
        labels = {
            "natural": "Escena natural",
            "urban": "Escena urbana o estructurada",
            "coastal": "Escena abierta con cielo o agua",
            "desert": "Escena arida o de terreno seco",
            "indoor": "Escena interior o neutra",
        }
        return labels.get(scene_name, "Escena general")

    def _suggestion_phrase(self, scene_name, confidence):
        if confidence > 1.2:
            prefix = "con bastante claridad"
        elif confidence > 0.5:
            prefix = "de forma moderada"
        else:
            prefix = "de forma tentativa"

        mapping = {
            "natural": prefix + " un paisaje abierto",
            "urban": prefix + " un entorno construido",
            "coastal": prefix + " un entorno exterior amplio",
            "desert": prefix + " una zona seca o despejada",
            "indoor": prefix + " un espacio interior o visualmente neutro",
        }
        return mapping.get(scene_name, prefix + " una escena general")
