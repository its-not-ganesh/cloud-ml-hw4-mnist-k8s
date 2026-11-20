import io
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 4_000_000


def uploaded_pixels(data):
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 4_000_000:
                raise ValueError("Image exceeds four million pixels")
            image.load()
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            gray = background.convert("L")
        values = np.asarray(gray)
        if int(values.max()) - int(values.min()) < 10:
            raise ValueError("Image contains no visible foreground")
        border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1]))
        if np.median(border) > 127:
            gray = ImageOps.invert(gray)
        mask = np.asarray(gray) > 40
        if not mask.any():
            raise ValueError("Image contains no visible foreground")
        ys, xs = np.where(mask)
        digit = gray.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
        result = Image.new("L", (28, 28))
        result.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
        return np.asarray(result, dtype=np.float32) / 255
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or oversized image") from exc
