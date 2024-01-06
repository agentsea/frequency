import os
from typing import List, Dict, Any

import runpod

from .base import InferenceEndpiont, InferenceProvider


class RunPodProvider(InferenceProvider):
    """A runpod infra provider"""

    def __init__(self) -> None:
        key = os.getenv("RUNPOD_API_KEY")
        if not key:
            raise ValueError("must set $RUNPOD_API_KEY")
        runpod.api_key = key

    def run(self, name: str, image: str, gpu_type: str) -> InferenceEndpiont:
        resp = runpod.create_pod(name, image, gpu_type)
        print("response from create: ", resp)

    def status(self, name: str) -> Dict[str, Any]:
        resp = runpod.get_pod(name)
        print("status: ", resp)
        return resp

    def running(self) -> List[str]:
        resp = runpod.get_pods()
        print("list response: ", resp)
        return resp

    def stop(self, name: str) -> None:
        runpod.stop_pod(name)
