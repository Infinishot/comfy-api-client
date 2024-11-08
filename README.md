# comfy-api-client

A Python client for the ComfyUI API, providing:

:zap: Full API coverage  
:zap: Asynchronous execution  
:zap: WebSocket support  
:zap: Workflow templating  
:zap: WebSocket or HTTP-based polling

## Installation

Install the package using `pip`:

```bash
pip install comfy-api-client
```

## Usage

### Create a Client

Use the `create_client` context manager to create a ComfyUI client. This will set up the underlying HTTP client and a WebSocket or HTTP-based state tracker to poll results from the server:

```python
from comfy_api_client import create_client

# Protocol is omitted as the URL may be used for both HTTP and WebSocket requests
comfyui_server = "localhost:8188"

async with create_client(comfyui_server) as client:
    print(await client.get_system_stats())
```

### Submit Workflows

To submit a workflow, read the workflow configuration file and pass it to the client:

```python
from comfy_api_client import utils

workflow = utils.read_json("workflow.json")

async with create_client(comfyui_server) as client:
    prompt = await client.submit_workflow(workflow)

    result = await prompt.future
    image_items = result.output_images
    image = image_items[0].image
```

## Result polling

### HTTP-based polling

By default, the execution state of a prompt is tracked via a WebSocket connection. In cases where a WebSocket connection cannot be established, e.g. if the ComfyUI server is behind a proxy or firewall, HTTP-based polling can be used instead.

Simply provide `start_state_tracker="http"` to the `create_client` function:

```python
async with create_client(comfyui_server, start_state_tracker="http") as client:
    # Submit and await results as before
    prompt = await client.submit_workflow(workflow)
    result = await prompt.future
```

Note, that HTTP polling relies on frequent querying of the prompt status from the ComfyUI server and will thus create more traffic while being less responsive.

### Manual polling

The prompt status and results can also be queried manually. In this case, `start_state_tracker=None` can be passed to `create_client` and `return_future=False` to the `submit_workflow()` method.

```python
import time

async with create_client(comfyui_server, start_state_tracker=None) as client:
    # Submit and await result as before
    prompt = await client.submit_workflow(workflow, return_future=False)

    # Will be None; the status needs to be checked manually
    prompt.future

    # Wait for the prompt to finish
    time.sleep(20)

    result = await client.fetch_results(prompt)

    image_items = result.output_images
    image = image_items[0].image
```

## Tests

Run tests:

```bash
pytest tests
```

This will set up a local ComfyUI instance to test against.

## TODOs

- [ ] Add logging support
- [ ] Improve error handling and messages
- [ ] Implement a synchronous client