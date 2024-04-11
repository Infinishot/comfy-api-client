import requests
import time
import io

from PIL.Image import Image


def check_connection(url, delay=0.5, timeout=10):
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(url)

            if response.status_code == 200:
                return True

            return False
        except requests.RequestException:
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay)

    return False


def image_to_buffer(image: Image, format="jpeg"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer
