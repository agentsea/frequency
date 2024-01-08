# Frequency

Efficiently serve LoRA tuned models.

Frequency provides a means to hot-swap LoRA layers in ML models at the time of inference allowing for the efficient usage of large base models.

## Install

```
pip install frequency-ai
```

Install server component on Kubernetes

```
helm install frequency oci://artifact.frequency.ai/frequency-server:0.0.1
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

# Cache an adapter on the server that was trained on dog images
resp = model.cache_adapter(name="dog", hf_repo="Anima-ai/dog_lora")

# Qwen expects a specific format for describing images
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
query = tokenizer.from_list_format([
    {'image': 'https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/pembroke-welsh-corgi.jpg'},
    {'text': 'What is this?'},
])

# Chat with the model using the dog adapter
response, history = model.chat(query=query, adapters=["dog"])
#> Here is a picture of a Corgi

# Cache an adapter on the server that was trained on cat images
resp = model.cache_adapter(name="cat", hf_repo="Anima-ai/cat_lora")
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
