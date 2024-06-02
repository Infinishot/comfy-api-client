import asyncio
from functools import wraps
import json
import random
import requests
import time
import io

from PIL.Image import Image
from urllib.parse import urlparse, ParseResult as URLParseResult
from comfy_api_client import constants


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


def retry_fn(fn, retries=3, delay=0.5, backoff=2):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if i == retries - 1:
                    raise

                time.sleep(delay * backoff**i)

    return wrapper


def async_retry_fn(fn, retries=3, delay=0.5, backoff=2):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        for i in range(retries):
            try:
                return await fn(*args, **kwargs)
            except Exception:
                if i == retries - 1:
                    raise

                await asyncio.sleep(delay * backoff**i)

    return wrapper


async def load_json_iter(it, ignore_non_string):
    async for data in it:
        if isinstance(data, str):
            yield json.loads(data)
        else:
            if not ignore_non_string:
                raise ValueError(f"Only string data is allowed but got {type(data)}")


def _replace_dict_values_recursively(
    dictionary: dict, target_key: list[str], new_value, start_at_root=False, copy=False
):
    if not target_key:
        return new_value

    if not isinstance(dictionary, dict):
        return dictionary

    if copy:
        dictionary = dict(dictionary)

    for key in dictionary.keys():
        if key == target_key[0]:
            dictionary[key] = _replace_dict_values_recursively(
                dictionary[key],
                target_key[1:],
                new_value,
                start_at_root=True,
                copy=copy,
            )
        elif not start_at_root:
            dictionary[key] = _replace_dict_values_recursively(
                dictionary[key],
                target_key,
                new_value,
                start_at_root=start_at_root,
                copy=copy,
            )

    return dictionary


def replace_dict_values(
    dictionary: dict, key: str | list[str], new_value, start_at_root=False, copy=False
):
    if isinstance(key, str):
        key = key.split(".")

    return _replace_dict_values_recursively(
        dictionary, key, new_value, copy=copy, start_at_root=start_at_root
    )


def replace_noise_seeds(workflow: dict, seed: int, noise_keys: list[str] | None = None):
    if noise_keys is None:
        noise_keys = constants.NOISE_KEYS

    for noise_key in noise_keys:
        workflow = replace_dict_values(
            workflow, ["inputs", noise_key], seed, copy=True, start_at_root=False
        )

    return workflow


def randomize_noise_seeds(workflow: dict, noise_keys: list[str] | None = None):
    seed = random.randint(0, 2**32 - 1)
    return replace_noise_seeds(workflow, seed, noise_keys=noise_keys)


def parse_url(url: str, default_scheme="http") -> URLParseResult:
    parsed = urlparse(url)
    
    if not parsed.netloc:
        return parse_url(f"{default_scheme}://{url}")
    
    return parsed