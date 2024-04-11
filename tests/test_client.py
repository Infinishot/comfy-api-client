import asyncio
from pathlib import Path
import pytest

from conftest import COMFY_INSTALL_LOCATION


@pytest.mark.asyncio
async def test_get_embeddings(comfyui_client, dummy_embedding):
    embeddings = await comfyui_client.get_embeddings()

    assert len(embeddings) == 1
    assert embeddings[0] == dummy_embedding.stem


@pytest.mark.asyncio
async def test_get_extensions(comfyui_client, dummy_embedding):
    extensions = await comfyui_client.get_extensions()

    assert len(extensions)
    assert all(ext.endswith(".js") for ext in extensions)


@pytest.mark.asyncio
async def test_upload_image(comfyui_client, dummy_image):
    response = await comfyui_client.upload_image("test.png", dummy_image)

    assert response.name == "test.png"
    assert response.subfolder == ""
    assert response.type == "input"


@pytest.mark.asyncio
async def test_upload_image_with_subfolder(comfyui_client, dummy_image):
    response = await comfyui_client.upload_image(
        "test.png", dummy_image, subfolder="images"
    )

    assert response.name == "test.png"
    assert response.subfolder == "images"


@pytest.mark.asyncio
async def test_upload_mask(comfyui_client, dummy_image):
    orig_ref = await comfyui_client.upload_image("test.png", dummy_image)
    response = await comfyui_client.upload_mask(
        "test.png", dummy_image, original_reference=orig_ref
    )

    assert response.name == "test.png"
    assert response.subfolder == ""
    assert response.type == "input"


@pytest.mark.asyncio
async def test_retrieve_unknown_image(comfyui_client):
    result = await comfyui_client.retrieve_image("missing.png")

    assert result is None


@pytest.mark.asyncio
async def test_retrieve_known_image(comfyui_client, dummy_image):
    await comfyui_client.upload_image("test.png", dummy_image, type="output")

    image_item = await comfyui_client.retrieve_image("test.png")

    assert image_item.filename == "test.png"
    assert image_item.format == "png"
    # assert utils.image_to_buffer(dummy_image).read() == utils.image_to_buffer(image_item.image).read()


@pytest.mark.asyncio
async def test_retrieve_meta_data(comfyui_client, dummy_embedding):
    dummy_embedding = dummy_embedding.relative_to(
        Path(COMFY_INSTALL_LOCATION) / "models"
    )

    metadata = await comfyui_client.retrieve_metadata(
        folder_name=str(dummy_embedding.parent), filename=str(dummy_embedding.name)
    )

    assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_get_system_stats(comfyui_client):
    stats = await comfyui_client.get_system_stats()

    assert len(stats.devices)


@pytest.mark.asyncio
async def test_get_empty_history(comfyui_client):
    history = await comfyui_client.get_history()

    assert not len(history)


@pytest.mark.asyncio
async def test_get_history(comfyui_client, dummy_workflow):
    prompt_info = await comfyui_client.enqueue_workflow(dummy_workflow)

    await asyncio.sleep(0.1)

    history = await comfyui_client.get_history()

    assert len(history) == 1
    assert prompt_info.prompt_id in history


@pytest.mark.asyncio
async def test_get_history_by_unknown_prompt(comfyui_client, dummy_workflow):
    history = await comfyui_client.get_history(prompt_id="unknown")

    assert len(history) == 0


@pytest.mark.asyncio
async def test_get_history_by_known_prompt(comfyui_client, dummy_workflow):
    prompt_info = await comfyui_client.enqueue_workflow(dummy_workflow)

    await asyncio.sleep(0.1)

    history = await comfyui_client.get_history(prompt_id=prompt_info.prompt_id)

    assert len(history) == 1
    assert prompt_info.prompt_id in history


@pytest.mark.asyncio
async def test_get_empty_queue(comfyui_client):
    queue = await comfyui_client.get_queue()

    assert not len(queue.pending)
    assert not len(queue.running)


@pytest.mark.skip(reason="Needs reconsideration")
@pytest.mark.asyncio
async def test_get_queue_pending(comfyui_client, dummy_workflow):
    await comfyui_client.free_memory()

    for _ in range(3):
        await comfyui_client.enqueue_workflow(dummy_workflow)

    # await asyncio.sleep(0.1)

    queue = await comfyui_client.get_queue()

    assert len(queue.pending)
    assert len(queue.running)


@pytest.mark.asyncio
async def test_enqueue_workflow(comfyui_client, dummy_workflow):
    prompt_info = await comfyui_client.enqueue_workflow(
        dummy_workflow, return_future=False
    )

    assert prompt_info.prompt_id


@pytest.mark.asyncio
async def test_enqueue_workflow_return_future(comfyui_client, dummy_workflow):
    loop = asyncio.get_event_loop()

    print(loop)
    prompt_info = await comfyui_client.enqueue_workflow(
        dummy_workflow, return_future=True
    )

    result = await prompt_info.future

    assert result.prompt_id == prompt_info.prompt_id
    assert len(result.output_images) == 1


@pytest.mark.skip(reason="Needs reconsideration")
@pytest.mark.asyncio
async def test_clear_queue(comfyui_client, dummy_workflow):
    for _ in range(3):
        await comfyui_client.enqueue_workflow(dummy_workflow)

    await comfyui_client.clear_queue()

    await asyncio.sleep(0.1)

    queue = await comfyui_client.get_queue()

    assert not len(queue.pending)
    assert not len(queue.running)
