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
prompt = await client.submit_workflow(workflow)

result = await prompt.future
```

## TODOs

- [ ] Add logging support
- [ ] Improve error handling and messages
- [ ] Implement a synchronous client