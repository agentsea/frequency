import sys
import time

sys.path.append("../frequency")

from frequency.client.v1.frequency_api import FrequencyAPI
from frequency.provider.runpod import RunPodProvider

GPU_ENABLED = False

MODEL = "facebook/opt-350m"
NAME = "opt"
LORA = "ybelkada/opt-350m-lora"

if GPU_ENABLED:
    MODEL = "Qwen/Qwen-VL-Chat"
    NAME = "qwen"
    LORA = ""

# provider = RunPodProvider()

# provider.run("test-freq", "")

# https://l88g599jicx8xx-8000.proxy.runpod.net/

# client = FrequencyAPI(endpoint="https://l88g599jicx8xx-8000.proxy.runpod.net/")
client = FrequencyAPI(endpoint="http://localhost:9090")

resp = client.get_health()
print("health: ", resp)

body = {
    "hf_repo": MODEL,
    "name": NAME,
    "type": "AutoModelForCausalLM",
    "cuda": GPU_ENABLED,
}
resp = client.load_model(body)
print("response from load model: ", resp)
time.sleep(5)

body = {
    "model": NAME,
    "hf_repo": LORA,
    "name": "adapter_1",
}
resp = client.load_adapter(body)
print("response from loading adapter: ", resp)

body = {
    "query": "Hello",
    "adapters": ["adapter_1"],
}
resp = client.generate(NAME, body)
print("response from chat: ", resp)

body = {
    "query": "Hello how are you?",
}
resp = client.generate(NAME, body)
print("response from chat: ", resp)


# resp = client.get_adapters()
# print("get adapters: ", resp)

# resp = client.get_models()
# print("get models: ", resp)
