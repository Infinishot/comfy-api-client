import abc
from asyncio import Future
import asyncio
from contextlib import asynccontextmanager
import json
import io
import urllib.parse
import warnings
import httpx
from typing import AsyncGenerator, Literal
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


# TODO: finish this
class HistoryItemSchema(BaseModel):
    prompt: tuple[int, str, dict, dict, list[str]]


class HistorySchema(RootModel):
    root: dict[str, HistoryItemSchema]


class ComfyUIAPIClient:
    def __init__(self, comfy_url: str, client: httpx.AsyncClient):
        self.comfy_url = utils.parse_url(comfy_url)
        
        if self.comfy_url.scheme not in ["http", "https"]:
            raise ValueError(f"Provided Comfy URL scheme is not supported: {self.comfy_url.scheme}")
        
        self.client = client

        self.client_id = uuid.uuid4().hex
        self.websocket_handler_task = None
        self.futures = {}

        self.housekeepers = []
        
    def endpoint(self, path: str | None = None):
        base_url = f"{self.comfy_url.scheme}://{self.comfy_url.netloc}"
        
        if path is None:
            return base_url
        
        return urllib.parse.urljoin(base_url, path)

    async def get_index(self):
        response = await self.client.get(self.endpoint())
        response.raise_for_status()

        return response.text

    @validate_call(validate_return=True)
    async def get_embeddings(self) -> FileList:
        response = await self.client.get(self.endpoint("embeddings"))
        response.raise_for_status()

        return response.json()

    @validate_call(validate_return=True)
    async def get_extensions(self) -> FileList:
        response = await self.client.get(self.endpoint("extensions"))
        response.raise_for_status()

        return response.json()

    def _prepare_image_upload(
        self, name: str, image: Image | str, subfolder: str | None = None
    ) -> ImageUploadResponse:
        if isinstance(image, str):
            image = ImageFactory.open(image)

        blob = utils.image_to_buffer(image, format="png")

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
            self.endpoint("/upload/image"),
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
            self.endpoint("/upload/mask"),
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
    ) -> ImageItem | None:
        params = {
            "filename": filename,
        }

        if subfolder:
            params["subfolder"] = subfolder

        if type:
            params["type"] = type

        response = await self.client.get(
            self.endpoint("/view"), params=params
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
            self.endpoint(f"/view_metadata/{folder_name_encoded}"),
            params={"filename": filename},
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        return response.json()

    async def get_system_stats(self) -> SystemStats:
        response = await self.client.get(self.endpoint("/system_stats"))
        response.raise_for_status()

        return SystemStats(**response.json())

    async def get_node_info(self) -> dict:
        raise NotImplementedError

    async def get_object_info(self) -> dict:
        raise NotImplementedError

    async def get_history(
        self, prompt_id: str | None = None, max_items: int | None = None
    ) -> dict:
        endpoint = self.endpoint("/history")

        if prompt_id:
            endpoint += f"/{prompt_id}"

        params = {}

        if max_items:
            params["max_items"] = max_items

        response = await self.client.get(endpoint, params=params)

        response.raise_for_status()

        return response.json()

    async def get_completed_prompts(self):
        history = await self.get_history()

        return {
            prompt_id: data
            for prompt_id, data in history.items()
            if data.get("status", {}).get("completed")
        }

    async def get_queue(self) -> QueueState:
        response = await self.client.get(self.endpoint("/queue"))
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

    async def fetch_results(self, prompt_id: str) -> PromptResult:
        output_images = []

        history = await self.get_history(prompt_id=prompt_id)

        assert prompt_id in history

        result = history[prompt_id]

        for output in result["outputs"].values():
            if "images" not in output:
                continue

            for image in output["images"]:
                image_item = await self.retrieve_image(
                    image["filename"],
                    subfolder=image["subfolder"],
                    type=image["type"],
                )

                output_images.append(image_item)

        return PromptResult(prompt_id=prompt_id, output_images=output_images)

    async def get_future(self, prompt_id: str):
        return self.futures.get(prompt_id)

    async def get_future_with_retry(self, prompt_id: str):
        get_future = utils.async_retry_fn(self.get_future)
        return await get_future(prompt_id)

    # @property

    async def stop_websocket_handler(self):
        if not self.is_websocket_handler_running:
            warnings.warn("Websocket handler is not running")
            return

        self.websocket_handler_task.cancel()

        try:
            await self.websocket_handler_task
        except asyncio.CancelledError:
            pass

    async def enqueue_workflow(
        self, workflow: dict, return_future: bool = True
    ) -> PromptResponse:
        if return_future:
            if not any([keeper.is_running for keeper in self.housekeepers]):
                warnings.warn(
                    "enqueue_workflow has been called with return_future=True, but no housekeeper is running. "
                    "Results might therefore not be received and have to be requested manually."
                )

            loop = asyncio.get_event_loop()
            future = loop.create_future()

        response = await self.client.post(
            self.endpoint("/prompt"),
            json={"prompt": workflow, "client_id": self.client_id},
        )
        response.raise_for_status()

        prompt_info = PromptResponse(**response.json())

        if return_future:
            self.futures[prompt_info.prompt_id] = future
            prompt_info.future = future

        return prompt_info

    async def clear_queue(self):
        response = await self.client.post(
            self.endpoint("/queue"), json={"clear": True}
        )
        response.raise_for_status()

    async def cancel_prompts(self, prompt_ids: list[str]):
        response = await self.client.post(
            self.endpoint("/queue"), json={"cancel": prompt_ids}
        )
        response.raise_for_status()

    async def interrupt_processing(self):
        response = await self.client.post(self.endpoint("/interrupt"))
        response.raise_for_status()

    async def free_memory(self, unload_models: bool = False, free_memory: bool = False):
        response = await self.client.post(
            self.endpoint("/free"),
            json={"unload_models": unload_models, "free_memory": free_memory},
        )
        response.raise_for_status()

    async def clear_history(self):
        response = await self.client.post(
            self.endpoint("/history"), json={"clear": True}
        )
        response.raise_for_status()

    async def remove_from_history(self, prompt_ids: list[str]):
        response = await self.client.post(
            self.endpoint("/history"), json={"delete": prompt_ids}
        )
        response.raise_for_status()

    def register_housekeeper(self, housekeeper: "BaseComfyClientHousekeeper"):
        self.housekeepers.append(housekeeper)

    def remove_housekeeper(self, housekeeper: "BaseComfyClientHousekeeper"):
        self.housekeepers.remove(housekeeper)


class BaseComfyClientHousekeeper(abc.ABC):
    def __init__(self, client: ComfyUIAPIClient):
        self.client = client
        self.client.register_housekeeper(self)

    @abc.abstractmethod
    async def start():
        pass

    @abc.abstractmethod
    async def stop():
        pass


class WebsocketComfyClientHousekeeper(BaseComfyClientHousekeeper):
    def __init__(
        self,
        client: ComfyUIAPIClient,
        use_secure_websocket: bool = False,
        **websocket_connect_kwargs,
    ):
        super().__init__(client)

        self.use_secure_websocket = use_secure_websocket
        self.run_task = None
        self.is_running = False
        self.websocket_connect_kwargs = websocket_connect_kwargs
        self.websocket = None

    def get_protocol(self):
        return "wss" if self.use_secure_websocket else "ws"

    async def run(self):
        websocket_connect_kwargs = {
            "max_size": 100 * 2**30,
            **self.websocket_connect_kwargs,
        }

        try:
            async with websockets.connect(
                f"{self.get_protocol()}://{self.client.comfy_url.netloc}/ws?clientId={self.client.client_id}",
                **websocket_connect_kwargs,
            ) as websocket:
                self.websocket = websocket

                async for message in utils.load_json_iter(
                    websocket, ignore_non_string=True
                ):
                    if message["type"] == "executing":
                        data = message["data"]
                        if data["node"] is None:
                            prompt_id = data["prompt_id"]

                            # The websocket might already return results before the enqueueing request
                            # has returned and the future has been assigned.
                            # We therefore query for the availability of the future with an exponential backoff.
                            future = await self.client.get_future_with_retry(prompt_id)

                            assert future is not None

                            try:
                                future.set_result(
                                    await self.client.fetch_results(prompt_id)
                                )
                            except Exception as e:
                                future.set_exception(e)
        finally:
            self.websocket = None

    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        self.run_task = asyncio.get_event_loop().create_task(self.run())

    async def stop(self):
        if self.websocket is None:
            warnings.warn("Websocket is not open")
        else:
            await self.websocket.close()

        await self.run_task

        self.is_running = False


class HTTPComfyClientHousekeeper(BaseComfyClientHousekeeper):
    def __init__(self, client: ComfyUIAPIClient, update_interval: float = 0.5):
        super().__init__(client)

        self.update_interval = update_interval
        self.is_running = False
        self.current_update_task = None
        self.thread = None

    async def update(self):
        if not self.is_running:
            return

        completed = await self.client.get_completed_prompts()

        for prompt_id, result in completed.items():
            if prompt_id in self.client.futures:
                future = await self.client.get_future(prompt_id)

                if future and not future.done():
                    try:
                        future.set_result(await self.client.fetch_results(prompt_id))
                    except Exception as e:
                        future.set_exception(e)

        await asyncio.sleep(self.update_interval)

        self.current_update_task = asyncio.get_event_loop().create_task(self.update())

    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        self.current_update_task = asyncio.get_event_loop().create_task(self.update())

        self.client.register_housekeeper(self)

    async def stop(self):
        self.is_running = False

        if self.current_update_task is not None:
            await self.current_update_task

        self.client.remove_housekeeper(self)


housekeeper_implementations = {
    "websocket": WebsocketComfyClientHousekeeper,
    "http": HTTPComfyClientHousekeeper,
}


async def create_comfy_client_housekeeper(client, name: str, **housekeeper_kwargs):
    cls = housekeeper_implementations.get(name)

    if cls is None:
        raise ValueError(
            f"Unknown Comfy client housekeeper '{name}'. Valid options are {list(housekeeper_implementations)}"
        )

    housekeeper = cls(client=client, **housekeeper_kwargs)
    await housekeeper.start()

    return housekeeper


@asynccontextmanager
async def create_client(
    comfy_url: str,
    start_housekeeper: Literal["websocket", "http"] | None = "websocket",
    housekeeper_kwargs: dict | None = None,
) -> AsyncGenerator[ComfyUIAPIClient, None]:
    async with httpx.AsyncClient() as client:
        client = ComfyUIAPIClient(comfy_url, client)

        if start_housekeeper is not None:
            await create_comfy_client_housekeeper(
                client, start_housekeeper, **(housekeeper_kwargs or {})
            )
        try:
            yield client
        finally:
            for housekeeper in client.housekeepers:
                await housekeeper.stop()
