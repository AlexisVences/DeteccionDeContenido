import struct
import zlib


class PNGReader:
    def __init__(self, path):
        self.path = path
        self.width = None
        self.height = None
        self.bit_depth = None
        self.color_type = None
        self.pixels = []

    def read(self):
        with open(self.path, "rb") as f:
            self._validate_signature(f)
            chunks = self._read_chunks(f)
            self._parse_ihdr(chunks["IHDR"])
            raw_data = self._decompress_idat(chunks["IDAT"])
            self.pixels = self._reconstruct_pixels(raw_data)

    def _validate_signature(self, f):
        signature = f.read(8)
        expected = b'\x89PNG\r\n\x1a\n'
        if signature != expected:
            raise ValueError("Not a valid PNG file")

    def _read_chunks(self, f):
        chunks = {}
        idat_data = b""

        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break

            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = f.read(4).decode()
            data = f.read(length)
            crc = f.read(4)

            if chunk_type == "IHDR":
                chunks["IHDR"] = data
            elif chunk_type == "IDAT":
                idat_data += data
            elif chunk_type == "IEND":
                break

        chunks["IDAT"] = idat_data
        return chunks

# ESTRUCTURA DEL CHUNK IHDR 
# Width        4 bytes
# Height       4 bytes
# Bit depth    1 byte
# Color type   1 byte
# Compression  1 byte
# Filter       1 byte
# Interlace    1 byte

    def _parse_ihdr(self, data):
        self.width = struct.unpack(">I", data[0:4])[0]
        self.height = struct.unpack(">I", data[4:8])[0]
        self.bit_depth = data[8]
        self.color_type = data[9]

        # Solo aceptar RGB y RGBA
        if self.color_type not in (2, 6):
            raise ValueError(f"Unsupported PNG color type: {self.color_type}")

    def _decompress_idat(self, data):
        return zlib.decompress(data)
    
    def _reconstruct_pixels(self, raw_data):
        bytes_per_pixel = 4 if self.color_type == 6 else 3
        stride = self.width * bytes_per_pixel

        pixels = []
        index = 0

        prev_row = [0] * stride

        for _ in range(self.height):
            filter_type = raw_data[index]
            index += 1

            row = list(raw_data[index:index + stride])
            index += stride

            reconstructed = [0] * stride

            if filter_type == 0:
                reconstructed = row

            elif filter_type == 1:  # Sub
                for i in range(stride):
                    left = reconstructed[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                    reconstructed[i] = (row[i] + left) % 256

            elif filter_type == 2:  # Up
                for i in range(stride):
                    up = prev_row[i]
                    reconstructed[i] = (row[i] + up) % 256

            elif filter_type == 3:  # Average
                for i in range(stride):
                    left = reconstructed[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                    up = prev_row[i]
                    avg = (left + up) // 2
                    reconstructed[i] = (row[i] + avg) % 256

            elif filter_type == 4:  # Paeth
                for i in range(stride):
                    left = reconstructed[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                    up = prev_row[i]
                    up_left = prev_row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0

                    p = left + up - up_left
                    pa = abs(p - left)
                    pb = abs(p - up)
                    pc = abs(p - up_left)

                    if pa <= pb and pa <= pc:
                        predictor = left
                    elif pb <= pc:
                        predictor = up
                    else:
                        predictor = up_left

                    reconstructed[i] = (row[i] + predictor) % 256

            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

            # Convertir a RGB si es RGBA
            if self.color_type == 6:
                rgb_row = []
                for i in range(0, len(reconstructed), 4):
                    r = reconstructed[i]
                    g = reconstructed[i + 1]
                    b = reconstructed[i + 2]
                    # ignoramos alpha
                    rgb_row.extend([r, g, b])
                pixels.append(rgb_row)
            else:
                pixels.append(reconstructed)

            prev_row = reconstructed

        return pixels