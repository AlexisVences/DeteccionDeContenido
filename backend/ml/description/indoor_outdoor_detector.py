class IndoorOutdoorDetector:

    def detect(self, semantic, color, edge, spatial):
        indoor_score = 0.0
        outdoor_score = 0.0
        evidence = []

        dominant_color = color.get("dominant_color", "mixed")
        top_blue_ratio = spatial.get("top_blue_ratio", 0)
        bottom_green_ratio = spatial.get("bottom_green_ratio", 0)
        edge_density = edge.get("edge_density", 0)
        colorfulness = color.get("colorfulness", 0)
        texture = semantic.get("texture", "")
        palette_hint = semantic.get("palette_hint", "")
        center_darkness_bias = spatial.get("center_darkness_bias", 0)

        if dominant_color in ("gray", "mixed"):
            indoor_score += 0.8
            evidence.append("paleta poco natural")

        if colorfulness < 20:
            indoor_score += 0.7
            evidence.append("variacion de color reducida")
        else:
            outdoor_score += 0.4

        if 0.16 <= edge_density <= 0.34:
            indoor_score += 0.8
            evidence.append("estructura con bordes de interior")

        if top_blue_ratio > 0.30:
            outdoor_score += 1.0
            evidence.append("apertura superior azulada")

        if top_blue_ratio > 0.18 and dominant_color == "blue":
            outdoor_score += 0.8

        if bottom_green_ratio > 0.20:
            outdoor_score += 0.9
            evidence.append("presencia inferior de vegetacion")

        if semantic.get("openness") == "abierta":
            outdoor_score += 0.8

        if texture in ("organica", "muy variada"):
            outdoor_score += 0.5

        if semantic.get("palette_hint") == "paleta neutra o urbana":
            indoor_score += 0.7

        if palette_hint == "predominancia de cielo o agua":
            outdoor_score += 0.9

        if dominant_color == "green":
            outdoor_score += 0.7

        if top_blue_ratio < 0.10 and bottom_green_ratio < 0.10 and colorfulness < 18:
            indoor_score += 0.6

        if edge_density > 0.30 and colorfulness > 18 and dominant_color in ("blue", "green"):
            outdoor_score += 0.5

        if dominant_color == "blue" and spatial.get("center_darkness_bias", 0) < -30:
            outdoor_score += 0.8
            evidence.append("sujeto central iluminado en escena abierta")

        if center_darkness_bias < -25 and edge_density > 0.22:
            outdoor_score += 0.7
            evidence.append("sujeto destacado en entorno exterior")

        if dominant_color == "blue" and top_blue_ratio > 0.10 and center_darkness_bias < -25:
            outdoor_score += 0.5

        label = "interior" if indoor_score >= outdoor_score else "exterior"

        return {
            "is_indoor": label == "interior",
            "label": label,
            "indoor_score": round(indoor_score, 3),
            "outdoor_score": round(outdoor_score, 3),
            "evidence": evidence[:4],
        }
