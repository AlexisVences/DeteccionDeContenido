from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.forensics.raw_image_reader import PNGReader
from backend.ml.description.image_descriptor import ImageDescriptor


def main():
    image_path = PROJECT_ROOT / "backend" / "ml" / "forensics" / "paisaje_natural.png"

    reader = PNGReader(str(image_path))
    reader.read()

    descriptor = ImageDescriptor()
    result = descriptor.describe_from_reader(reader)

    print(result["description"])


if __name__ == "__main__":
    main()
