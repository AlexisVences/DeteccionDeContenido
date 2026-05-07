class AdvancedSemanticFeatureExtractor:

    def extract(self, feature_dict):
        dominant_color = feature_dict.get("dominant_color", "mixed")

        return {
            "brightness": self._brightness(feature_dict["mean"]),
            "contrast": self._contrast(feature_dict["variance"]),
            "complexity": self._complexity(feature_dict["entropy"]),
            "detail": self._detail(feature_dict["gradiente_mean"]),
            "texture": self._texture(
                feature_dict["block_energy_variance"],
                feature_dict.get("edge_density", 0),
            ),
            "edge_profile": self._edge_profile(feature_dict.get("edge_density", 0)),
            "dominant_color": dominant_color,
            "palette_hint": self._palette_hint(
                dominant_color,
                feature_dict.get("colorfulness", 0),
                feature_dict.get("top_blue_ratio", 0),
                feature_dict.get("edge_density", 0),
            ),
            "spatial_layout": self._spatial_layout(
                feature_dict.get("top_blue_ratio", 0),
                feature_dict.get("bottom_green_ratio", 0),
                feature_dict.get("bottom_yellow_ratio", 0),
            ),
            "openness": self._openness(
                feature_dict.get("top_blue_ratio", 0),
                feature_dict.get("edge_density", 0),
            ),
        }

    def _brightness(self, mean_value):
        if mean_value < 80:
            return "oscura"
        if mean_value < 160:
            return "balanceada"
        return "brillante"

    def _contrast(self, variance_value):
        if variance_value < 500:
            return "bajo contraste"
        if variance_value < 2000:
            return "contraste medio"
        return "alto contraste"

    def _complexity(self, entropy_value):
        if entropy_value < 3:
            return "simple"
        if entropy_value < 5:
            return "moderada"
        return "compleja"

    def _detail(self, gradient_mean):
        if gradient_mean < 10:
            return "suave"
        if gradient_mean < 30:
            return "con detalles moderados"
        return "rica en detalles"

    def _texture(self, block_var, edge_density):
        if block_var < 50 and edge_density < 0.10:
            return "uniforme"
        if block_var < 200:
            return "mixta"
        if edge_density < 0.18:
            return "organica"
        return "muy variada"

    def _edge_profile(self, edge_density):
        if edge_density < 0.12:
            return "bordes suaves"
        if edge_density < 0.24:
            return "bordes moderados"
        return "bordes densos"

    def _palette_hint(self, dominant_color, colorfulness, top_blue_ratio, edge_density):
        if dominant_color == "green":
            return "predominancia de vegetacion"
        if dominant_color == "blue" and (top_blue_ratio > 0.28 or edge_density < 0.18):
            return "predominancia de cielo o agua"
        if dominant_color == "blue":
            return "paleta fria azulada"
        if dominant_color == "yellow":
            return "predominancia de tonos calidos o arena"
        if dominant_color == "gray":
            return "paleta neutra o urbana"
        if colorfulness > 35:
            return "paleta variada"
        return "paleta mixta"

    def _spatial_layout(self, top_blue_ratio, bottom_green_ratio, bottom_yellow_ratio):
        if top_blue_ratio > 0.35 and bottom_green_ratio > 0.20:
            return "cielo arriba y vegetacion abajo"
        if top_blue_ratio > 0.35 and bottom_yellow_ratio > 0.20:
            return "cielo arriba y suelo calido abajo"
        if top_blue_ratio > 0.35:
            return "franja superior abierta"
        return "distribucion sin patron fuerte"

    def _openness(self, top_blue_ratio, edge_density):
        if top_blue_ratio > 0.30 and edge_density < 0.18:
            return "abierta"
        if edge_density > 0.28:
            return "estructurada"
        return "intermedia"
