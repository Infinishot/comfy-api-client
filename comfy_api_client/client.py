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
from comfy_api_client.utils import PathLike


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


class ComfyExecutionError(Exception):
    pass


class ComfyAPIClient:
    """Asynchronous client for the ComfyUI backend API.
    
    Args:
        comfy_url (str): URL of the ComfyUI instance.
        client (httpx.AsyncClient): Async HTTP client to use for requests.
    """
    
    def __init__(self, comfy_url: str, client: httpx.AsyncClient):
        self.comfy_url = utils.parse_url(comfy_url)
        
        if self.comfy_url.scheme not in ["http", "https"]:
            raise ValueError(f"Provided Comfy URL scheme is not supported: {self.comfy_url.scheme}")
        
        self.client = client

        self.client_id = uuid.uuid4().hex
        self.websocket_handler_task = None
        self.futures = {}

        self.state_trackers = []
        
    def get_endpoint_url(self, path: str | None = None):
        """Get the full URL for an endpoint.
        
        Args:
            path (str, optional): Endpoint path. Defaults to "/".
        """
        base_url = f"{self.comfy_url.scheme}://{self.comfy_url.netloc}"
        
        if path is None:
            return base_url
        
        return urllib.parse.urljoin(base_url, path)

    async def get_index(self):
        """Get the index page of the ComfyUI instance."""
        response = await self.client.get(self.get_endpoint_url())
        response.raise_for_status()

        return response.text

    @validate_call(validate_return=True)
    async def get_embeddings(self) -> FileList:
        """Get a list of available embeddings.
        
        Returns:
            FileList: List of available embeddings.
            
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self.client.get(self.get_endpoint_url("embeddings"))
        response.raise_for_status()

        return response.json()

    @validate_call(validate_return=True)
    async def get_extensions(self) -> FileList:
        """Get a list of available extensions.
        
        Returns:
            A list of available extensions as a FileList object.
        
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self.client.get(self.get_endpoint_url("extensions"))
        response.raise_for_status()

        return response.json()

    def _prepare_image_upload(
        self, name: str, image: Image | PathLike, subfolder: str | None = None
    ) -> ImageUploadResponse:
        if isinstance(image, PathLike):
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
        image: Image | PathLike,
        subfolder: str | None = None,
        overwrite: bool = True,
        type: ImageTypes = "input",
    ) -> ImageUploadResponse:
        """Upload an image to the ComfyUI instance.
        
        Args:
            name (str): Name of the image.
            image (Image | str | Path): Image to upload. Can be a PIL Image or a path to an image file.
            subfolder (str, optional): Subfolder to upload the image to. Defaults to no subfolder.
            overwrite (bool, optional): Whether to overwrite an existing image with the same name. Defaults to True.
            type (ImageTypes, optional): Type of the image. Defaults to "input".
            
        Returns:
            ImageUploadResponse: Information about the uploaded image
        
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self.client.post(
            self.get_endpoint_url("/upload/image"),
            data={"overwrite": overwrite, "type": type},
            files=self._prepare_image_upload(name, image, subfolder),
        )
        response.raise_for_status()

        return ImageUploadResponse(**response.json())

    async def upload_mask(
        self,
        name: str,
        image: Image | PathLike,
        original_reference: ImageUploadResponse,
        subfolder: str | None = None,
        overwrite: bool = True,
        type: ImageTypes = "input",
    ) -> ImageUploadResponse:
        """Upload a mask image to the ComfyUI instance.
        
        Args:
            name (str): Name of the image.
            image (Image | str | Path): Image to upload. Can be a PIL Image or a path to an image file.
            original_reference (ImageUploadResponse): Reference to the original image.
            subfolder (str, optional): Subfolder to upload the image to. Defaults to no subfolder.
            overwrite (bool, optional): Whether to overwrite an existing image with the same name. Defaults to True.
            type (ImageTypes, optional): Type of the image. Defaults to "input".
        
        Returns:
            ImageUploadResponse: Information about the uploaded image
            
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        original_reference = original_reference.model_dump()
        original_reference["filename"] = original_reference.pop("name")

        response = await self.client.post(
            self.get_endpoint_url("/upload/mask"),
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
        """Retrieve an image from the ComfyUI instance.
        
        Args:
            filename (str): Name of the image.
            subfolder (str, optional): Subfolder to retrieve the image from. Defaults to no subfolder.
            type: (ImageTypes, optional): Type of the image. Defaults to None.
        
        Returns:
            ImageItem: Image item containing the image, filename and format.
        
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        params = {
            "filename": filename,
        }

        if subfolder:
            params["subfolder"] = subfolder

        if type:
            params["type"] = type

        response = await self.client.get(
            self.get_endpoint_url("/view"), params=params
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
        """Retrieve metadata for a file in a folder.
        
        Args:
            folder_name (str): Name of the folder.
            filename (str): Name of the file.
            
        Returns:
            dict: Metadata for the file.
        
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        folder_name_encoded = urllib.parse.quote_plus(folder_name)

        response = await self.client.get(
            self.get_endpoint_url(f"/view_metadata/{folder_name_encoded}"),
            params={"filename": filename},
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        return response.json()

    async def get_system_stats(self) -> SystemStats:
        """Get system statistics.
        
        Returns:
            SystemStats: System statistics.
        
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self.client.get(self.get_endpoint_url("/system_stats"))
        response.raise_for_status()

        return SystemStats(**response.json())

    async def get_node_info(self) -> dict:
        raise NotImplementedError

    async def get_object_info(self) -> dict:
        raise NotImplementedError

    async def get_history(
        self, prompt_id: str | None = None, max_items: int | None = None
    ) -> dict:
        """Get the history of prompts.
        
        TODO: Implement proper schema for history.
        
        Args:
            prompt_id (str, optional): ID of the prompt to retrieve. Defaults to None in which case information about all prompts is returned.
            max_items (int, optional): Maximum number of items to return. Defaults to None.
        
        Returns:
            dict: History of prompts.
        
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        endpoint = self.get_endpoint_url("/history")

        if prompt_id:
            endpoint += f"/{prompt_id}"

        params = {}

        if max_items:
            params["max_items"] = max_items

        response = await self.client.get(endpoint, params=params)

        response.raise_for_status()

        return response.json()

    async def get_completed_prompts(self):
        """Get a list of completed prompts.
        
        Returns:
            dict: Completed prompts in the form {<prompt_id>: <prompt_data>}.
            
            Raises:"""
        history = await self.get_history()

        return {
            prompt_id: data
            for prompt_id, data in history.items()
            if data.get("status", {}).get("completed")
        }

    async def get_queue(self) -> QueueState:
        """Get the current queue state.
        
        Returns:
            QueueState: Queue state.
            
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self.client.get(self.get_endpoint_url("/queue"))
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

    async def fetch_results(self, prompt: str | PromptResponse) -> PromptResult:
        """Fetch the results of a prompt.
        
        Args:
            prompt (str | Prompt): ID of the prompt or Prompt object.
            
        Returns:
            PromptResult: Results of the prompt.
            
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        if isinstance(prompt, PromptResponse):
            prompt_id = prompt.prompt_id
        elif isinstance(prompt, str):
            prompt_id = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
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

    async def _get_future(self, prompt_id: str):
        return self.futures.get(prompt_id)

    async def _get_future_with_retry(self, prompt_id: str):
        get_future = utils.async_retry_fn(self._get_future)
        return await get_future(prompt_id)

    async def submit_workflow(
        self, workflow: dict, return_future: bool = True
    ) -> PromptResponse:
        """Enqueue a workflow for execution.
        
        TODO: Implement proper schema for workflow.
        
        Args:
            workflow (dict): Workflow to enqueue.
            return_future (bool, optional): Whether to return a future that will be set with the results of the prompt. Defaults to True.
        
        Returns:
            PromptResponse: Information about the prompt.
            
        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        if return_future:
            if not any([tracker.is_running for tracker in self.state_trackers]):
                warnings.warn(
                    "enqueue_workflow has been called with return_future=True, but no state tracker is running. "
                    "Results might therefore not be received and have to be requested manually."
                )

            loop = asyncio.get_event_loop()
            future = loop.create_future()

        response = await self.client.post(
            self.get_endpoint_url("/prompt"),
            json={"prompt": workflow, "client_id": self.client_id},
        )
        response.raise_for_status()

        prompt_info = PromptResponse(**response.json())

        if return_future:
            self.futures[prompt_info.prompt_id] = future
            prompt_info.future = future

        return prompt_info

    async def clear_queue(self) -> None:
        """Clear the prompt queue."""
        response = await self.client.post(
            self.get_endpoint_url("/queue"), json={"clear": True}
        )
        response.raise_for_status()

    async def cancel_prompts(self, prompt_ids: list[str]) -> None:
        """Cancel prompts in the queue.
        
        Args:
            prompt_ids (list[str]): List of prompt IDs to cancel.
        """
        response = await self.client.post(
            self.get_endpoint_url("/queue"), json={"cancel": prompt_ids}
        )
        response.raise_for_status()

    async def interrupt_processing(self) -> None:
        response = await self.client.post(self.get_endpoint_url("/interrupt"))
        response.raise_for_status()

    async def free_memory(self, unload_models: bool = False, free_memory: bool = False) -> None:
        response = await self.client.post(
            self.get_endpoint_url("/free"),
            json={"unload_models": unload_models, "free_memory": free_memory},
        )
        response.raise_for_status()

    async def clear_history(self) -> None:
        response = await self.client.post(
            self.get_endpoint_url("/history"), json={"clear": True}
        )
        response.raise_for_status()

    async def remove_from_history(self, prompt_ids: list[str]) -> None:
        response = await self.client.post(
            self.get_endpoint_url("/history"), json={"delete": prompt_ids}
        )
        response.raise_for_status()

    def register_state_tracker(self, state_tracker: "BaseComfyStateTracker") -> None:
        self.state_trackers.append(state_tracker)

    def remove_state_tracker(self, state_tracker: "BaseComfyStateTracker") -> None:
        self.state_trackers.remove(state_tracker)


class BaseComfyStateTracker(abc.ABC):
    """Base class for Comfy state trackers.
    
    A state tracker is responsible for tracking the state of prompts and their results and updating 
    futures in the client accordingly.
    """
    def __init__(self, client: ComfyAPIClient):
        self.client = client

    @abc.abstractmethod
    async def start():
        pass

    @abc.abstractmethod
    async def stop():
        pass


class WebsocketStateTracker(BaseComfyStateTracker):
    """State tracker that uses a websocket to track prompt state.
    
    Args:
        client (ComfyAPIClient): Comfy client to keep track of.
        use_secure_websocket (bool, optional): Whether to use a secure websocket. Defaults to False.
        **websocket_connect_kwargs: Additional keyword arguments to pass to the websocket connect function.
    """
    def __init__(
        self,
        client: ComfyAPIClient,
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
        self.client.register_state_tracker(self)
        
        if self.is_running:
            return

        self.is_running = True
        self.run_task = asyncio.get_event_loop().create_task(self.run())

    async def stop(self):
        self.client.remove_state_tracker(self)
        
        if self.websocket is None:
            warnings.warn("Websocket is not open")
        else:
            await self.websocket.close_connection()

        await self.run_task

        self.is_running = False


class HTTPStateTracker(BaseComfyStateTracker):
    """State tracker that uses the REST API to track prompt state.
    
    Args:
        client (ComfyAPIClient): Comfy client to keep track of.
        update_interval (float, optional): Interval in seconds to update the prompt state. Defaults to 0.1.
    """
    def __init__(self, client: ComfyAPIClient, update_interval: float = 0.1):
        super().__init__(client)

        self.update_interval = update_interval
        self.is_running = False
        self.current_update_task = None
        self.thread = None

    async def update(self):
        if not self.is_running:
            return

        history = await self.client.get_history()

        for prompt_id, result in history.items():
            if prompt_id in self.client.futures:
                future = await self.client._get_future(prompt_id)

                if future and not future.done():
                    status = result["status"]
                    status_msg = status.get("status_str")
                    
                    if status_msg == "success":
                        try:
                            future.set_result(await self.client.fetch_results(prompt_id))
                        except Exception as e:
                            future.set_exception(e)
                    elif status_msg == "error":
                        future.set_exception(ComfyExecutionError(f"Prompt failed: {json.dumps(status)}"))

        await asyncio.sleep(self.update_interval)

        self.current_update_task = asyncio.get_event_loop().create_task(self.update())

    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        self.current_update_task = asyncio.get_event_loop().create_task(self.update())

        self.client.register_state_tracker(self)

    async def stop(self):
        self.is_running = False

        if self.current_update_task is not None:
            await self.current_update_task

        self.client.remove_state_tracker(self)


state_tracker_implementations = {
    "websocket": WebsocketStateTracker,
    "http": HTTPStateTracker,
}


async def create_comfy_state_tracker(client, name: str, **state_tracker_kwargs):
    """Create a Comfy state tracker.
    
    Args:
        client (ComfyAPIClient): Comfy client to keep track of.
        name (str): Name of the state tracker to create.
        **state_tracker_kwargs: Additional keyword arguments to pass to the state tracker constructor.
    """
    cls = state_tracker_implementations.get(name)

    if cls is None:
        raise ValueError(
            f"Unknown Comfy client state tracker '{name}'. Valid options are {list(state_tracker_implementations)}"
        )

    state_tracker = cls(client=client, **state_tracker_kwargs)
    await state_tracker.start()

    return state_tracker


@asynccontextmanager
async def create_client(
    comfy_url: str,
    http_timeout: float | None = 30.0,
    start_state_tracker: Literal["websocket", "http"] | None = "websocket",
    state_tracker_kwargs: dict | None = None,
) -> AsyncGenerator[ComfyAPIClient, None]:
    """Create a ComfyUI client.
    
    Args:
        comfy_url (str): URL of the ComfyUI server.
        http_timeout (float, optional): Timeout for HTTP requests. Defaults to 30.0.
        start_state_tracker (Literal["websocket", "http"], optional): State tracker to start. Defaults to "websocket".
        state_tracker_kwargs (dict, optional): Additional keyword arguments to pass to the state tracker constructor.
    """
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        client = ComfyAPIClient(comfy_url, client)
        
        await client.get_index()

        if start_state_tracker is not None:
            await create_comfy_state_tracker(
                client, start_state_tracker, **(state_tracker_kwargs or {})
            )
        try:
            yield client
        finally:
            for state_tracker in client.state_trackers:
                await state_tracker.stop()
