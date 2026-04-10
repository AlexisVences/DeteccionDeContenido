from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.forensics.raw_image_reader import PNGReader
from backend.ml.description.image_descriptor import ImageDescriptor


def run_descriptor(image_relative_path):
    image_path = PROJECT_ROOT / image_relative_path

    reader = PNGReader(str(image_path))
    reader.read()

    descriptor = ImageDescriptor()
    result = descriptor.describe_from_reader(reader)

    print(f"\nImagen: {image_path.name}")
    print("Descripcion:")
    print(result["description"])
    print("\nEscena inferida:")
    print(result["scene_inference"]["top_scene"])
    print(result["scene_inference"]["scores"])
    print("\nSemantica:")
    print(result["semantic_features"])
    print("\nHeuristicas:")
    print(result["heuristic_features"])


def main():
    run_descriptor(Path("backend/ml/forensics/paisaje.png"))
    run_descriptor(Path("backend/ml/forensics/negro.png"))


if __name__ == "__main__":
    main()
