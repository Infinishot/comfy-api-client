from contextlib import contextmanager
import json
import sys
import subprocess
import pytest
import pytest_asyncio
import psutil
import numpy as np
from httpx import AsyncClient
from pathlib import Path
from PIL import Image
from safetensors.numpy import save_file
from comfy_api_client import utils
from comfy_api_client.client import create_client


COMFY_INSTALL_LOCATION = "comfyui"
COMFY_URL = "localhost:8188"


def install_comfyui():
    p = subprocess.Popen(
        ["/bin/sh"],
        text=True,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    p.stdin.write(
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY_INSTALL_LOCATION}\n"
    )
    p.stdin.write(f"cd {COMFY_INSTALL_LOCATION}\n")
    p.stdin.write("python -m venv .venv\n")
    p.stdin.write(". .venv/bin/activate\n")
    p.stdin.write(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
    )
    p.stdin.write("pip install -r requirements.txt\n")
    p.stdin.write("pip install torchsde\n")
    
    # Remove ComfyUI's git folder
    # Otherwise the folder may appear as subrepository
    p.stdin.write("rm -rf .git\n")
    
    p.communicate()
    p.wait()


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.terminate()
    process.terminate()


@pytest.fixture(scope="session", autouse=True)
def comfyui():
    if not Path(COMFY_INSTALL_LOCATION).exists():
        install_comfyui()

    p = subprocess.Popen(
        ["/bin/sh"],
        shell=True,
        text=True,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        p.stdin.write(f"cd {COMFY_INSTALL_LOCATION}\n")
        p.stdin.write(". .venv/bin/activate\n")
        p.stdin.write("python main.py --cpu --verbose\n")
        p.stdin.flush()

        if not utils.check_connection("http://" + COMFY_URL, timeout=10):
            raise Exception("Could not connect to ComfyUI server.")

        yield
    finally:
        kill(p.pid)


@contextmanager
def create_dummy_model_file(subdir, name, touch_file: bool = True):
    model_path = Path(COMFY_INSTALL_LOCATION) / subdir / name

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

        yield model_path
    finally:
        model_path.unlink(missing_ok=True)


@pytest.fixture()
def dummy_embedding():
    with create_dummy_model_file(
        "models/embeddings", "embedding.safetensors", touch_file=False
    ) as path:
        save_file(
            {"embedding": np.random.randn(128)}, filename=path, metadata={"foo": "bar"}
        )
        yield path


@pytest.fixture()
def dummy_image():
    return Image.open("tests/fixtures/image.png")


@pytest.fixture()
def random_image():
    return Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))


@pytest.fixture()
def dummy_workflow():
    return json.loads(Path("tests/fixtures/dummy_workflow.json").read_text())


@pytest_asyncio.fixture
async def httpx_client():
    async with AsyncClient() as client:
        yield client


@pytest_asyncio.fixture
async def comfyui_client():
    async with create_client(COMFY_URL) as client:
        yield client


@pytest_asyncio.fixture(autouse=True)
async def upload_dummy_input_image(comfyui_client, dummy_image):
    await comfyui_client.upload_image("input.png", dummy_image)
    yield "input.png"
