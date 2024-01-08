import os
from typing import List, Dict, Any, Optional

import runpod

from .base import InferenceEndpiont, InferenceProvider


class RunPodProvider(InferenceProvider):
    """A runpod infra provider"""

    def __init__(self) -> None:
        key = os.getenv("RUNPOD_API_KEY")
        if not key:
            raise ValueError("must set $RUNPOD_API_KEY")
        runpod.api_key = key

    def run(
        self,
        name: str,
        image: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        gpu_cpu_count: Optional[int] = None,
        hf_repo: Optional[str] = None,
    ) -> InferenceEndpiont:
        resp = runpod.create_pod(name, image, gpu_type, ports=["8000/tcp"])
        print("response from create: ", resp)

        endpoint = f"https://{name}-8000.proxy.runpod.net"
        return InferenceEndpiont(endpoint)

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
