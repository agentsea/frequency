# Frequency

Efficiently serve LoRA tuned models.

Frequency provides a means to hot-swap LoRA layers in ML models at the time of inference allowing for the efficient usage of large base models.

## Install

```
pip install frequency-ai
```

Install server component on Kubernetes

```
helm install oci://frequency.ai/frequency-server:0.0.1
```

## Usage

Load a HuggingFace model and use adapters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from frequency import Client

# Connect to the frequency server
client = Client("localhost:9000")

# Load an hf model onto the server
model = client.load_model(name="qwen-vl-chat", hf_repo="Qwen/Qwen-VL-Chat", type=AutoModelForCausalLM)
print(resp)

# Cache an adapter on the server that was trained on dog images
resp = model.cache_adapter(name="dog", uri="gs://my-adapters/dog_lora.pt")
print(resp)

# Query the model with the hot swap adapter
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

query = tokenizer.from_list_format([
    {'image': 'https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/pembroke-welsh-corgi.jpg'},
    {'text': 'What is this?'},
])

# Chat with the model using the dog adapter
response, history = model.chat(query=query, adapters=["dog"])
#> Here is a picture of a Corgi

response, history = model.chat(query="Output the detection frame of the dog's head", adapters=["dog"], history=history)
print(response)
#> <ref>Dog head/ref><box>(517,508),(589,611)</box>

image = tokenizer.draw_bbox_on_latest_picture(response, history)

if image:
  image.save('head.jpg')

# Cache an adapter on the server that was trained on cat images
resp = model.cache_adapter(name="cat", uri="gs://my-adapters/cat_lora.pt")
print(resp)

query = tokenizer.from_list_format([
    {'image': 'https://www.catster.com/wp-content/uploads/2023/11/Brown-tabby-cat-that-curls-up-outdoors_viper-zero_Shutterstock-800x533.jpg'},
    {'text': 'What is this?'},
])

# Chat with the same model using the new cat adapter
response, history = model.chat(query=query, adapters=["cat"])
#> Here is a picture of a tabby cat
```

## Roadmap

- Tenancy
