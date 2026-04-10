class SceneScoringEngine:

    def score(self, semantic_features, heuristic_features, model_predictions=None):
        scores = {
            "natural": 0.0,
            "urban": 0.0,
            "coastal": 0.0,
            "desert": 0.0,
            "indoor": 0.0,
        }
        evidence = {scene_name: [] for scene_name in scores}

        self._apply_semantic_scores(scores, evidence, semantic_features)
        self._apply_heuristic_scores(scores, evidence, heuristic_features)
        self._apply_model_scores(scores, evidence, model_predictions)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_scene, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0
        confidence = top_score - second_score

        return {
            "scores": scores,
            "ranked_scenes": ordered,
            "top_scene": top_scene,
            "confidence": confidence,
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
            evidence["coastal"].append("luminosidad alta")

        if texture in ("muy variada", "orgánica"):
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

        for scene_name in scores:
            scores[scene_name] = round(scores[scene_name], 3)

    def _apply_model_scores(self, scores, evidence, model_predictions):
        if not model_predictions:
            return

        for scene_name, weight in model_predictions.items():
            if scene_name in scores:
                scores[scene_name] += weight
                evidence[scene_name].append("prior del modelo")
