from asyncio import Future
import asyncio
from contextlib import asynccontextmanager
import json
import io
import urllib.parse
import httpx

from typing import ContextManager, Literal
import uuid
from pydantic import BaseModel, RootModel, validate_call

from PIL.Image import Image
from PIL import Image as ImageFactory
import websockets

from comfy_api_client import utils


class ListLikeMixin:
    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class FileList(ListLikeMixin, RootModel[list[str]]):
    root: list[str]


ImageTypes = Literal["input", "temp", "output"]


class ImageUploadResponse(BaseModel):
    name: str
    subfolder: str
    type: ImageTypes


class ImageItem(BaseModel, arbitrary_types_allowed=True):
    image: Image
    filename: str
    format: str


class SystemInfo(BaseModel):
    os: str
    python_version: str
    embedded_python: bool


class DeviceInfo(BaseModel):
    name: str
    type: str
    index: int | None
    vram_total: int
    torch_vram_total: int
    torch_vram_free: int


class SystemStats(BaseModel):
    system: SystemInfo
    devices: list[DeviceInfo]


class PromptResponse(BaseModel, arbitrary_types_allowed=True):
    prompt_id: str
    number: int
    node_errors: dict
    future: Future | None = None


class QueueEntry(BaseModel):
    number: int
    prompt_id: str
    prompt: dict
    extra_data: dict
    outputs_to_execute: list[str]


class QueueState(BaseModel):
    pending: list[QueueEntry]
    running: list[QueueEntry]


class PromptResult(BaseModel):
    prompt_id: str
    output_images: list[ImageItem]


class ComfyUIAPIClient:
    def __init__(self, comfy_host: str, client: httpx.AsyncClient):
        self.comfy_host = comfy_host
        self.client = client

    async def get_index(self):
        response = await self.client.get(f"http://{self.comfy_host}")
        response.raise_for_status()

        return response.text

    @validate_call(validate_return=True)
    async def get_embeddings(self) -> FileList:
        response = await self.client.get(f"http://{self.comfy_host}/embeddings")
        response.raise_for_status()

        return response.json()

    @validate_call(validate_return=True)
    async def get_extensions(self) -> FileList:
        response = await self.client.get(f"http://{self.comfy_host}/extensions")
        response.raise_for_status()

        return response.json()

    def _prepare_image_upload(
        self, name: str, image: Image | str, subfolder: str | None = None
    ) -> ImageUploadResponse:
        if isinstance(image, str):
            image = ImageFactory.open(image)

        blob = utils.image_to_buffer(image)

        files = {
            "image": (name, blob, "image/png"),
            "overwrite": (None, "true"),
        }

        if subfolder:
            files["subfolder"] = (None, subfolder)

        return files

    async def upload_image(
        self,
        name: str,
        image: Image | str,
        subfolder: str | None = None,
        overwrite: bool = True,
        type: ImageTypes = "input",
    ) -> ImageUploadResponse:
        response = await self.client.post(
            f"http://{self.comfy_host}/upload/image",
            data={"overwrite": overwrite, "type": type},
            files=self._prepare_image_upload(name, image, subfolder),
        )
        response.raise_for_status()

        return ImageUploadResponse(**response.json())

    async def upload_mask(
        self,
        name: str,
        image: Image | str,
        original_reference: ImageUploadResponse,
        subfolder: str | None = None,
        overwrite: bool = True,
        type: ImageTypes = "input",
    ) -> ImageUploadResponse:
        original_reference = original_reference.model_dump()
        original_reference["filename"] = original_reference.pop("name")

        response = await self.client.post(
            f"http://{self.comfy_host}/upload/mask",
            data={
                "overwrite": overwrite,
                "type": type,
                "original_ref": json.dumps(original_reference),
            },
            files=self._prepare_image_upload(name, image, subfolder),
        )
        response.raise_for_status()

        return ImageUploadResponse(**response.json())

    async def retrieve_image(
        self,
        filename: str,
        subfolder: str | None = None,
        type: ImageTypes | None = None,
    ) -> Image | None:
        params = {
            "filename": filename,
        }

        if subfolder:
            params["subfolder"] = subfolder

        if type:
            params["type"] = type

        response = await self.client.get(
            f"http://{self.comfy_host}/view", params=params
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        image = ImageFactory.open(io.BytesIO(response.content))

        filename = response.headers["Content-Disposition"].split("=")[1][1:-1]
        format = response.headers["Content-Type"].split("/")[1]

        return ImageItem(
            image=image,
            filename=filename,
            format=format,
        )

    async def retrieve_metadata(self, folder_name: str, filename: str) -> dict | None:
        folder_name_encoded = urllib.parse.quote_plus(folder_name)

        response = await self.client.get(
            f"http://{self.comfy_host}/view_metadata/{folder_name_encoded}",
            params={"filename": filename},
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        return response.json()

    async def get_system_stats(self) -> SystemStats:
        response = await self.client.get(f"http://{self.comfy_host}/system_stats")
        response.raise_for_status()

        return SystemStats(**response.json())

    async def get_node_info(self) -> dict:
        raise NotImplementedError

    async def get_object_info(self) -> dict:
        raise NotImplementedError

    async def get_history(
        self, prompt_id: str | None = None, max_items: int | None = None
    ) -> dict:
        endpoint = f"http://{self.comfy_host}/history"

        if prompt_id:
            endpoint += f"/{prompt_id}"

        params = {}

        if max_items:
            params["max_items"] = max_items

        response = await self.client.get(endpoint, params=params)

        response.raise_for_status()

        return response.json()

    async def get_queue(self) -> QueueState:
        response = await self.client.get(f"http://{self.comfy_host}/queue")
        response.raise_for_status()

        data = response.json()

        def parse_queue_entry(entry: tuple) -> QueueEntry:
            number, prompt_id, prompt, extra_data, outputs_to_execute = entry

            return QueueEntry(
                number=number,
                prompt_id=prompt_id,
                prompt=prompt,
                extra_data=extra_data,
                outputs_to_execute=outputs_to_execute,
            )

        return QueueState(
            pending=list(map(parse_queue_entry, data["queue_pending"])),
            running=list(map(parse_queue_entry, data["queue_running"])),
        )

    async def _poll_prompt(self, client_id, future):
        output_images = []
        prompt_id = None

        try:
            async with websockets.connect(
                f"ws://{self.comfy_host}/ws?clientId={client_id}"
            ) as websocket:
                async for message in websocket:
                    if isinstance(message, str):
                        message = json.loads(message)
                        if message["type"] == "executing":
                            data = message["data"]
                            if data["node"] is None:
                                prompt_id = data["prompt_id"]
                                break

            assert prompt_id is not None

            history = await self.get_history(prompt_id=prompt_id)

            assert prompt_id in history

            result = history[prompt_id]

            for output in result["outputs"].values():
                for image in output["images"]:
                    image_item = await self.retrieve_image(
                        image["filename"],
                        subfolder=image["subfolder"],
                        type=image["type"],
                    )

                    output_images.append(image_item)

        except Exception as e:
            future.set_exception(e)
            return

        future.set_result(
            PromptResult(prompt_id=prompt_id, output_images=output_images)
        )

    async def enqueue_workflow(
        self, workflow: dict, return_future: bool = True
    ) -> PromptResponse:
        client_id = uuid.uuid4().hex

        if return_future:
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            asyncio.create_task(self._poll_prompt(client_id, future))

        response = await self.client.post(
            f"http://{self.comfy_host}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        response.raise_for_status()

        prompt_info = PromptResponse(**response.json())

        if return_future:
            prompt_info.future = future

        return prompt_info

    async def clear_queue(self):
        response = await self.client.post(
            f"http://{self.comfy_host}/queue", json={"clear": True}
        )
        response.raise_for_status()

    async def cancel_prompts(self, prompt_ids: list[str]):
        response = await self.client.post(
            f"http://{self.comfy_host}/queue", json={"cancel": prompt_ids}
        )
        response.raise_for_status()

    async def interrupt_processing(self):
        response = await self.client.post(f"http://{self.comfy_host}/interrupt")
        response.raise_for_status()

    async def free_memory(self, unload_models: bool = False, free_memory: bool = False):
        response = await self.client.post(
            f"http://{self.comfy_host}/free",
            json={"unload_models": unload_models, "free_memory": free_memory},
        )
        response.raise_for_status()

    async def clear_history(self):
        response = await self.client.post(
            f"http://{self.comfy_host}/history", json={"clear": True}
        )
        response.raise_for_status()

    async def remove_from_history(self, prompt_ids: list[str]):
        response = await self.client.post(
            f"http://{self.comfy_host}/history", json={"delete": prompt_ids}
        )
        response.raise_for_status()


@asynccontextmanager
async def create_client(comfy_host: str) -> ContextManager[ComfyUIAPIClient]:
    async with httpx.AsyncClient() as client:
        yield ComfyUIAPIClient(comfy_host, client)
