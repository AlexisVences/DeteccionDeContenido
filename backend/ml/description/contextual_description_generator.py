class ContextualDescriptionGenerator:

    def generate(
        self,
        semantic_features,
        scene_result,
        heuristic_features,
        face_result,
        object_context,
        indoor_result,
    ):
        parts = [self._scene_sentence(semantic_features, scene_result)]

        face_sentence = self._face_sentence(face_result, indoor_result)
        if face_sentence:
            parts.append(face_sentence)

        context_sentence = self._context_sentence(object_context, indoor_result)
        if context_sentence:
            parts.append(context_sentence)

        if heuristic_features.get("evidence"):
            parts.append(
                "Senales observadas: " +
                ", ".join(heuristic_features["evidence"][:4]) +
                "."
            )

        return " ".join(parts)

    def _scene_sentence(self, semantic_features, scene_result):
        label = self._scene_label(scene_result["top_scene"])
        phrase = (
            f"{label} con {semantic_features['palette_hint']}, "
            f"{semantic_features['edge_profile']} y textura {semantic_features['texture']}"
        )

        if semantic_features["spatial_layout"] != "distribucion sin patron fuerte":
            phrase += f", con {semantic_features['spatial_layout']}"

        return phrase + "."

    def _face_sentence(self, face_result, indoor_result):
        if not face_result or not face_result.get("has_face"):
            return ""

        environment = ""
        if indoor_result:
            if indoor_result.get("is_indoor"):
                environment = " en un entorno interior"
            else:
                environment = " en un entorno exterior"

        return (
            "Se detecta un rostro humano con iluminacion "
            f"{face_result['face_brightness']} y tono de piel "
            f"{face_result['skin_tone']}{environment}."
        )

    def _context_sentence(self, object_context, indoor_result):
        observations = []

        if indoor_result:
            if indoor_result.get("is_indoor"):
                observations.append("escena interior")
            else:
                observations.append("escena exterior")

        if object_context:
            if object_context.get("has_road"):
                observations.append("posible carretera")
            if object_context.get("has_vehicle"):
                observations.append("presencia probable de vehiculo")
            if object_context.get("has_sea"):
                observations.append("presencia probable de mar")
            if not observations and object_context.get("observations"):
                observations.extend(object_context["observations"][:1])

        if not observations:
            return ""

        return "Contexto inferido: " + ", ".join(observations) + "."

    def _scene_label(self, scene_name):
        labels = {
            "natural": "Escena natural",
            "urban": "Escena urbana",
            "coastal": "Escena costera",
            "desert": "Escena arida o de terreno seco",
            "indoor": "Escena interior o neutra",
            "portrait": "Escena centrada en persona",
            "road": "Escena de via o carretera",
            "vehicle_scene": "Escena con contexto vehicular",
        }
        return labels.get(scene_name, "Escena general")
