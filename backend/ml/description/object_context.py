class ObjectContextDetector:

    def detect(self, features, color, spatial, edge):
        scores = {
            "vehicle": 0.0,
            "road": 0.0,
            "sea": 0.0,
        }
        evidence = {name: [] for name in scores}

        dominant_color = color.get("dominant_color", "mixed")
        top_blue_ratio = features.get("top_blue_ratio", 0)
        bottom_green_ratio = features.get("bottom_green_ratio", 0)
        edge_density = edge.get("edge_density", 0)
        horizontal_bias = features.get("horizontal_edge_bias", 0)
        block_var = features.get("block_energy_variance", 0)
        center_darkness_bias = features.get("center_darkness_bias", 0)
        center_brightness = features.get("center_brightness", 0)
        side_brightness = features.get("side_brightness", 0)
        colorfulness = color.get("colorfulness", 0)

        if dominant_color == "blue":
            scores["sea"] += 0.6
            evidence["sea"].append("predominio cromatico azul")

        if top_blue_ratio > 0.42:
            scores["sea"] += 1.0
            evidence["sea"].append("franja azul en la zona superior")

        if block_var < 120 and edge_density < 0.20:
            scores["sea"] += 0.6
            evidence["sea"].append("textura relativamente uniforme")

        if dominant_color == "gray":
            scores["road"] += 0.4
            scores["vehicle"] += 0.3
            evidence["road"].append("paleta gris oscura")
            evidence["vehicle"].append("paleta neutra")

        if 0.22 <= edge_density <= 0.38:
            scores["road"] += 0.5
        if edge_density > 0.30 and dominant_color in ("gray", "mixed"):
            scores["vehicle"] += 0.6
            evidence["road"].append("bordes estructurados")
            evidence["vehicle"].append("contornos definidos")

        if horizontal_bias > 1.12:
            scores["road"] += 0.9
        if horizontal_bias > 1.18 and dominant_color in ("gray", "mixed"):
            scores["vehicle"] += 0.3
            evidence["road"].append("gradiente horizontal dominante")

        if center_darkness_bias > 10 and center_brightness < side_brightness:
            scores["road"] += 1.0
            evidence["road"].append("centro mas oscuro que los laterales")

        if (
            center_darkness_bias < -25 and
            edge_density > 0.24 and
            block_var > 1000000
        ):
            scores["vehicle"] += 1.1
            evidence["vehicle"].append("objeto central estructurado")

        if center_darkness_bias < -45 and edge_density > 0.26:
            scores["vehicle"] += 0.4
            evidence["vehicle"].append("contraste central compatible con vehiculo")

        if (
            edge_density > 0.30 and
            block_var > 220 and
            dominant_color in ("gray", "mixed")
        ):
            scores["vehicle"] += 0.7
            evidence["vehicle"].append("alta complejidad local")

        if bottom_green_ratio > 0.20:
            scores["vehicle"] -= 0.7
            scores["road"] -= 0.4

        if dominant_color == "green":
            scores["vehicle"] -= 0.8
            scores["road"] -= 0.5

        if edge_density > 0.26:
            scores["sea"] -= 0.5

        if center_darkness_bias > 15:
            scores["sea"] -= 0.7

        if block_var > 500000:
            scores["sea"] -= 0.4

        if colorfulness < 12 and dominant_color == "gray":
            scores["sea"] -= 0.6

        if dominant_color not in ("blue", "mixed"):
            scores["sea"] -= 0.5

        scores["vehicle"] = max(0.0, scores["vehicle"])
        scores["road"] = max(0.0, scores["road"])
        scores["sea"] = max(0.0, scores["sea"])

        result = {
            "vehicle_score": round(scores["vehicle"], 3),
            "road_score": round(scores["road"], 3),
            "sea_score": round(scores["sea"], 3),
            "has_vehicle": scores["vehicle"] >= 1.4,
            "has_road": scores["road"] >= 1.7,
            "has_sea": scores["sea"] >= 1.5,
            "evidence": evidence,
        }

        result["observations"] = self._observations(result)
        return result

    def _observations(self, result):
        observations = []
        if result["has_vehicle"]:
            observations.append("posible vehiculo")
        if result["has_road"]:
            observations.append("posible carretera")
        if result["has_sea"]:
            observations.append("posible mar")
        return observations
