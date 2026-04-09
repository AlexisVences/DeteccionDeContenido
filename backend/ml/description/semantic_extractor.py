# backend/ml/description/semantic_extractor.py

class SemanticFeatureExtractor:
    def extract(self, feature_dict):
        """
        Convierte features numéricas en atributos semánticos.
        """
        return {
            "brightness": self._brightness(feature_dict["mean"]),
            "contrast": self._contrast(feature_dict["variance"]),
            "complexity": self._complexity(feature_dict["entropy"]),
            "detail": self._detail(feature_dict["gradiente_mean"]),
            "texture": self._texture(feature_dict["block_energy_variance"])
        }

    def _brightness(self, mean):
        if mean < 80:
            return "oscura"
        elif mean < 160:
            return "balanceada"
        return "brillante"

    def _contrast(self, variance):
        if variance < 500:
            return "bajo contraste"
        elif variance < 2000:
            return "contraste medio"
        return "alto contraste"

    def _complexity(self, entropy):
        if entropy < 3:
            return "simple"
        elif entropy < 5:
            return "moderada"
        return "compleja"

    def _detail(self, gradient_mean):
        if gradient_mean < 10:
            return "suave"
        elif gradient_mean < 30:
            return "con detalles moderados"
        return "rica en detalles"

    def _texture(self, block_var):
        if block_var < 50:
            return "uniforme"
        elif block_var < 200:
            return "mixta"
        return "muy variada"