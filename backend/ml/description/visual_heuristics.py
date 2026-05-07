class VisualHeuristicAnalyzer:

    def analyze(self, semantic_features, raw_features):
        evidence = []

        dominant_color = raw_features.get("dominant_color", "mixed")
        top_blue_ratio = raw_features.get("top_blue_ratio", 0)
        bottom_green_ratio = raw_features.get("bottom_green_ratio", 0)
        bottom_yellow_ratio = raw_features.get("bottom_yellow_ratio", 0)
        edge_density = raw_features.get("edge_density", 0)
        block_variance = raw_features.get("block_energy_variance", 0)
        colorfulness = raw_features.get("colorfulness", 0)
        brightness_balance = raw_features.get("brightness_balance", 0)

        hints = {
            "palette_hint": self._palette_hint(dominant_color, colorfulness),
            "spatial_hint": self._spatial_hint(
                top_blue_ratio,
                bottom_green_ratio,
                bottom_yellow_ratio,
                brightness_balance,
            ),
            "structure_hint": self._structure_hint(
                edge_density,
                block_variance,
                semantic_features["detail"],
            ),
            "environment_flags": self._environment_flags(
                dominant_color,
                top_blue_ratio,
                bottom_green_ratio,
                bottom_yellow_ratio,
                edge_density,
                block_variance,
            ),
        }

        if dominant_color in ("green", "blue", "yellow", "gray", "red", "mixed"):
            evidence.append(
                f"color dominante {self._translate_color_name(dominant_color)}"
            )

        if top_blue_ratio > 0.35:
            evidence.append("zona superior con predominio azul")

        if bottom_green_ratio > 0.28:
            evidence.append("zona inferior con presencia de vegetacion")

        if bottom_yellow_ratio > 0.35 and dominant_color != "green":
            evidence.append("zona inferior con tonos calidos o arenosos")

        if edge_density > 0.28:
            evidence.append("alta densidad de bordes")
        elif edge_density < 0.12:
            evidence.append("bordes suaves")

        if block_variance > 250:
            evidence.append("variacion local elevada")
        elif block_variance < 60:
            evidence.append("estructura espacial uniforme")

        hints["evidence"] = evidence
        return hints

    def _translate_color_name(self, color_name):
        translations = {
            "green": "verde",
            "blue": "azul",
            "yellow": "amarillo",
            "gray": "gris",
            "red": "rojo",
            "mixed": "mixto",
        }
        return translations.get(color_name, color_name)

    def _palette_hint(self, dominant_color, colorfulness):
        if dominant_color == "green":
            return "predominancia de vegetacion"
        if dominant_color == "blue":
            return "predominancia de cielo o agua"
        if dominant_color == "yellow":
            return "predominancia de tonos calidos o arena"
        if dominant_color == "gray":
            return "paleta neutra o urbana"
        if colorfulness > 35:
            return "paleta variada"
        return "paleta mixta"

    def _spatial_hint(
        self,
        top_blue_ratio,
        bottom_green_ratio,
        bottom_yellow_ratio,
        brightness_balance,
    ):
        if top_blue_ratio > 0.35 and bottom_green_ratio > 0.20:
            return "cielo arriba y vegetacion abajo"

        if top_blue_ratio > 0.35 and bottom_yellow_ratio > 0.20:
            return "cielo arriba y terreno seco abajo"

        if brightness_balance > 15:
            return "parte superior mas luminosa"

        if brightness_balance < -15:
            return "parte inferior mas luminosa"

        return "distribucion espacial equilibrada"

    def _structure_hint(self, edge_density, block_variance, detail_label):
        if edge_density > 0.28 and block_variance > 220:
            return "estructura marcada y fragmentada"

        if edge_density < 0.12 and detail_label in ("suave", "con detalles moderados"):
            return "estructura suave y abierta"

        if block_variance > 180:
            return "estructura heterogenea"

        return "estructura moderada"

    def _environment_flags(
        self,
        dominant_color,
        top_blue_ratio,
        bottom_green_ratio,
        bottom_yellow_ratio,
        edge_density,
        block_variance,
    ):
        return {
            "nature_like": (
                dominant_color == "green" or
                bottom_green_ratio > 0.25 or
                (dominant_color == "blue" and top_blue_ratio > 0.35 and edge_density < 0.18)
            ),
            "urban_like": dominant_color == "gray" and edge_density > 0.22,
            "coast_like": (
                top_blue_ratio > 0.45 and
                dominant_color == "blue" and
                edge_density < 0.18 and
                block_variance < 500000
            ),
            "desert_like": dominant_color == "yellow" or bottom_yellow_ratio > 0.35,
            "open_scene_like": top_blue_ratio > 0.30 and edge_density < 0.20,
            "complex_scene_like": block_variance > 220 and edge_density > 0.24,
        }
