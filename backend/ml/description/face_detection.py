class FaceHeuristicDetector:

    def detect(self, pixels, width, height):
        """
        Retorna una estimacion heuristica de presencia de rostro.
        """
        skin_pixels = []
        weighted_skin = 0.0
        total_pixels = max(1, width * height)
        global_brightness_total = 0.0

        for y, row in enumerate(pixels):
            for x in range(0, len(row), 3):
                r = row[x]
                g = row[x + 1]
                b = row[x + 2]
                gray = (0.299 * r) + (0.587 * g) + (0.114 * b)
                global_brightness_total += gray

                if self._is_skin_like(r, g, b):
                    pixel_x = x // 3
                    skin_pixels.append((pixel_x, y, r, g, b, gray))
                    weighted_skin += self._center_weight(pixel_x, y, width, height)

        skin_ratio = len(skin_pixels) / total_pixels
        global_brightness = global_brightness_total / total_pixels

        if not skin_pixels:
            return {
                "has_face": False,
                "face_score": 0.0,
                "skin_tone": "indeterminado",
                "face_brightness": "indeterminada",
                "skin_ratio": 0.0,
            }

        bbox_score = self._bbox_score(skin_pixels, width, height)
        concentration_score = min(1.0, weighted_skin / max(1.0, len(skin_pixels)))
        face_score = (
            (min(1.0, skin_ratio / 0.18) * 0.45) +
            (bbox_score * 0.35) +
            (concentration_score * 0.20)
        )

        skin_tone = self._skin_tone(skin_pixels, global_brightness)
        face_brightness = self._face_brightness(skin_pixels, global_brightness)
        has_face = 0.12 <= skin_ratio <= 0.65 and face_score >= 0.38

        return {
            "has_face": has_face,
            "face_score": round(face_score, 3),
            "skin_tone": skin_tone,
            "face_brightness": face_brightness,
            "skin_ratio": round(skin_ratio, 3),
        }

    def _is_skin_like(self, r, g, b):
        return (
            r > g > b and
            80 <= r <= 255 and
            40 <= g <= 220 and
            20 <= b <= 180 and
            (r - g) >= 10 and
            (r - b) >= 20
        )

    def _center_weight(self, x, y, width, height):
        center_x = width / 2
        center_y = height / 2
        norm_x = abs(x - center_x) / max(1, center_x)
        norm_y = abs(y - center_y) / max(1, center_y)
        return max(0.2, 1.0 - ((norm_x + norm_y) / 2))

    def _bbox_score(self, skin_pixels, width, height):
        xs = [item[0] for item in skin_pixels]
        ys = [item[1] for item in skin_pixels]
        bbox_width = (max(xs) - min(xs) + 1) / max(1, width)
        bbox_height = (max(ys) - min(ys) + 1) / max(1, height)
        aspect_ratio = bbox_width / (bbox_height + 1e-6)

        size_score = min(1.0, (bbox_width * bbox_height) / 0.20)

        if 0.45 <= aspect_ratio <= 1.4:
            shape_score = 1.0
        elif 0.25 <= aspect_ratio <= 1.8:
            shape_score = 0.6
        else:
            shape_score = 0.2

        return (size_score * 0.6) + (shape_score * 0.4)

    def _skin_tone(self, skin_pixels, global_brightness):
        avg_luma = sum(item[5] for item in skin_pixels) / len(skin_pixels)
        adjusted = avg_luma - ((global_brightness - avg_luma) * 0.25)

        if adjusted < 95:
            return "oscura"
        if adjusted < 155:
            return "media"
        return "clara"

    def _face_brightness(self, skin_pixels, global_brightness):
        local_mean = sum(item[5] for item in skin_pixels) / len(skin_pixels)
        blended = (local_mean * 0.7) + (global_brightness * 0.3)

        if blended < 85:
            return "baja"
        if blended < 155:
            return "media"
        return "alta"
