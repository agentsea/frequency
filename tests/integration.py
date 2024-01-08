import sys

sys.path.append("../frequency")

from frequency.client.v1.frequency_api import FrequencyAPI
from frequency.provider.runpod import RunPodProvider

# provider = RunPodProvider()

# provider.run("test-freq", "")


client = FrequencyAPI(endpoint="http://localhost:8000")

resp = client.get_health()
print("health: ", resp)

body = {
    "hf_repo": "facebook/opt-350m",
    "name": "opt",
    "type": "AutoModelForCausalLM",
    "cuda": False,
}
resp = client.load_model(body)
print("response from load model: ", resp)

body = {
    "model": "opt",
    "hf_repo": "ybelkada/opt-350m-lora",
    "name": "adapter_1",
}
resp = client.load_adapter(body)
print("response from loading adapter: ", resp)

body = {
    "query": "Hello",
    "adapters": ["adapter_1"],
}
resp = client.chat_model("opt", body)
print("response from chat: ", resp)


resp = client.get_adapters()
print("get adapters: ", resp)

resp = client.get_models()
print("get models: ", resp)
