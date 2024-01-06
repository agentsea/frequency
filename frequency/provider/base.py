from abc import ABC, abstractmethod
from typing import Dict, Any, List


class TuningResult:
    uri: str


class TuningProvider(ABC):
    """An tuning infra provider"""

    @abstractmethod
    def tune(self, hf_repo: str) -> TuningResult:
        pass


class InferenceEndpiont:
    endpoint: str


class InferenceProvider(ABC):
    """An inference infra provider"""

    @abstractmethod
    def run(self, name: str, image: str, gpu_type: str) -> InferenceEndpiont:
        pass

    @abstractmethod
    def status(self, name: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def running(self) -> List[str]:
        pass

    @abstractmethod
    def stop(self, name: str) -> None:
        pass
