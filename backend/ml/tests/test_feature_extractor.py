import os
from ..forensics.raw_image_reader import PNGReader
from ..features.feature_extractor import FeatureExtractor

def process_folder(folder_path):

    results = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".png"):

            full_path = os.path.join(folder_path, filename)

            try:
                reader = PNGReader(full_path)
                reader.read()

                features = FeatureExtractor.extract_from_png_reader(reader)

                results.append({
                    "file": filename,
                    "features": features
                })

            except Exception as e:
                print(f"Error procesando {filename}: {e}")

    return results

results = process_folder("backend/ml/test_images")

for item in results:
    print(f"\nImagen: {item['file']}")
    for k, v in item["features"].items():
        print(f"{k}: {v}")