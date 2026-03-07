from ..forensics.raw_image_reader import PNGReader
from ..features.feature_extractor import FeatureExtractor

reader = PNGReader("backend/ml/test_images/paisaje.png")

reader.read()

features = FeatureExtractor.extract_from_png_reader(reader)

for k, v in features.items():
    print(f"{k}: {v}")
    
#Para ejecutar python -m backend.ml.tests.test_feature_extractor