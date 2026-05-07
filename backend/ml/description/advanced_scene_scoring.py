class AdvancedSceneScoringEngine:

    def score(
        self,
        semantic_features,
        heuristic_features,
        face_result=None,
        object_context=None,
        indoor_result=None,
        model_predictions=None,
    ):
        scores = {
            "natural": 0.0,
            "urban": 0.0,
            "coastal": 0.0,
            "desert": 0.0,
            "indoor": 0.0,
            "portrait": 0.0,
            "road": 0.0,
            "vehicle_scene": 0.0,
        }
        evidence = {scene_name: [] for scene_name in scores}

        self._apply_semantic_scores(scores, evidence, semantic_features)
        self._apply_heuristic_scores(scores, evidence, heuristic_features)
        self._apply_face_scores(scores, evidence, face_result)
        self._apply_context_scores(scores, evidence, object_context)
        self._apply_indoor_scores(scores, evidence, indoor_result)
        self._apply_model_scores(scores, evidence, model_predictions)

        for scene_name in scores:
            scores[scene_name] = round(max(0.0, scores[scene_name]), 3)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_scene, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0

        return {
            "scores": scores,
            "ranked_scenes": ordered,
            "top_scene": top_scene,
            "confidence": round(top_score - second_score, 3),
            "evidence": evidence[top_scene],
            "all_evidence": evidence,
        }

    def _apply_semantic_scores(self, scores, evidence, semantic_features):
        brightness = semantic_features["brightness"]
        texture = semantic_features["texture"]
        detail = semantic_features["detail"]
        palette = semantic_features["palette_hint"]

        if brightness == "brillante":
            scores["natural"] += 1.0
            scores["coastal"] += 0.8
            evidence["natural"].append("luminosidad alta")

        if texture in ("muy variada", "organica"):
            scores["natural"] += 0.8
            scores["urban"] += 0.4
            evidence["natural"].append("textura variada")

        if detail == "rica en detalles":
            scores["urban"] += 0.8
            scores["natural"] += 0.4
            evidence["urban"].append("alto nivel de detalle")

        if palette == "predominancia de vegetacion":
            scores["natural"] += 1.6
            evidence["natural"].append("dominancia cromatica verde")

        if palette == "predominancia de cielo o agua":
            scores["coastal"] += 1.4
            scores["natural"] += 0.8
            evidence["coastal"].append("dominancia cromatica azul")

        if palette == "predominancia de tonos calidos o arena":
            scores["desert"] += 1.7
            evidence["desert"].append("dominancia cromatica arenosa")

        if palette == "paleta neutra o urbana":
            scores["urban"] += 1.3
            scores["indoor"] += 0.7
            evidence["urban"].append("paleta gris o neutra")

    def _apply_heuristic_scores(self, scores, evidence, heuristic_features):
        spatial_hint = heuristic_features["spatial_hint"]
        structure_hint = heuristic_features["structure_hint"]
        flags = heuristic_features["environment_flags"]

        if spatial_hint == "cielo arriba y vegetacion abajo":
            scores["natural"] += 2.0
            evidence["natural"].append("composicion cielo-tierra")

        if spatial_hint == "cielo arriba y terreno seco abajo":
            scores["desert"] += 1.8
            scores["coastal"] += 0.6
            evidence["desert"].append("composicion cielo-suelo seco")

        if structure_hint == "estructura marcada y fragmentada":
            scores["urban"] += 1.6
            evidence["urban"].append("estructura intensa y fragmentada")

        if structure_hint == "estructura suave y abierta":
            scores["natural"] += 1.0
            scores["coastal"] += 0.8
            evidence["natural"].append("bordes suaves y apertura espacial")

        if flags["nature_like"]:
            scores["natural"] += 1.1

        if flags["urban_like"]:
            scores["urban"] += 1.2

        if flags["coast_like"]:
            scores["coastal"] += 1.3

        if flags["desert_like"]:
            scores["desert"] += 1.2

        if flags["open_scene_like"]:
            scores["coastal"] += 0.7
            scores["natural"] += 0.6

        if flags["complex_scene_like"]:
            scores["urban"] += 0.8
            scores["natural"] += 0.3

        if flags["complex_scene_like"] and not flags["nature_like"]:
            scores["urban"] += 0.5
            scores["natural"] -= 0.4

    def _apply_face_scores(self, scores, evidence, face_result):
        if not face_result:
            return

        if face_result.get("has_face"):
            face_score = face_result.get("face_score", 0)
            skin_ratio = face_result.get("skin_ratio", 0)
            scores["portrait"] += 1.8 + face_score + min(0.8, skin_ratio)
            scores["indoor"] += 0.5
            evidence["portrait"].append("rostro heuristico detectado")

            if face_result.get("face_brightness") == "baja":
                scores["indoor"] += 0.3
                evidence["indoor"].append("rostro con iluminacion baja")

            if skin_ratio > 0.25:
                scores["portrait"] += 0.5
                evidence["portrait"].append("region de piel significativa")

            scores["desert"] -= 0.6
            scores["coastal"] -= 0.4

    def _apply_context_scores(self, scores, evidence, object_context):
        if not object_context:
            return

        if object_context.get("has_sea"):
            scores["coastal"] += 1.8
            scores["natural"] += 0.4
            evidence["coastal"].append("indicios de mar")

        if object_context.get("has_road"):
            scores["road"] += 1.6
            scores["urban"] += 0.7
            evidence["road"].append("indicios de carretera")

        if object_context.get("has_vehicle"):
            scores["vehicle_scene"] += 2.4
            scores["urban"] += 0.5
            evidence["vehicle_scene"].append("indicios de vehiculo")

        if object_context.get("has_vehicle") and object_context.get("has_road"):
            scores["vehicle_scene"] += 1.1
            scores["road"] += 0.6
            evidence["vehicle_scene"].append("vehiculo sobre via probable")

        if object_context.get("has_vehicle") and not object_context.get("has_sea"):
            scores["coastal"] -= 0.4
            scores["natural"] -= 0.2

        if not object_context.get("has_sea"):
            scores["coastal"] -= 0.3

        if not object_context.get("has_vehicle"):
            scores["vehicle_scene"] -= 0.2

    def _apply_indoor_scores(self, scores, evidence, indoor_result):
        if not indoor_result:
            return

        if indoor_result.get("is_indoor"):
            scores["indoor"] += 1.5
            evidence["indoor"].append("patron visual de interior")
            scores["coastal"] -= 0.3
        else:
            scores["natural"] += 0.4
            scores["coastal"] += 0.3

    def _apply_model_scores(self, scores, evidence, model_predictions):
        if not model_predictions:
            return

        for scene_name, weight in model_predictions.items():
            if scene_name in scores:
                scores[scene_name] += weight
                evidence[scene_name].append("prior del modelo")
