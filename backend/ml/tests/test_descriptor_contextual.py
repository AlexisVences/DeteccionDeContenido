from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.forensics.raw_image_reader import PNGReader
from backend.ml.description.image_descriptor import ImageDescriptor


def describe_image(relative_path):
    image_path = PROJECT_ROOT / relative_path
    reader = PNGReader(str(image_path))
    reader.read()

    descriptor = ImageDescriptor()
    result = descriptor.describe_from_reader(reader)

    print(f"\nImagen: {image_path.name}")
    print("Face detection:")
    print(result["face_detection"])
    print("\nContext:")
    print(result["object_context"])
    print("\nIndoor/Outdoor:")
    print(result["indoor_outdoor"])
    print("\nScene final:")
    print(result["scene_inference"])
    print("\nDescription:")
    print(result["description"])


def main():
    test_images = [
        Path("backend/ml/forensics/paisaje_natural.png"),
        Path("backend/ml/forensics/paisaje_playa.png"),
        Path("backend/ml/forensics/paisaje_automovil.png"),
        Path("backend/ml/forensics/paisaje_urbano.png"),
        Path("backend/test_images/imagenes/real_and_fake_face/training_real/real_00975.png"),
    ]

    for image_path in test_images:
        if image_path.exists():
            describe_image(image_path)


if __name__ == "__main__":
    main()
