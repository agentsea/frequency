from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


class TuningResult:
    uri: str


class TuningProvider(ABC):
    """An tuning infra provider"""

    @abstractmethod
    def tune(self, hf_repo: str) -> TuningResult:
        pass


@dataclass
class InferenceEndpiont:
    endpoint: str


class InferenceProvider(ABC):
    """An inference infra provider"""

    @abstractmethod
    def run(
        self,
        name: str,
        image: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_memory: Optional[int] = None,
        gpu_cpu_count: Optional[int] = None,
        hf_repo: Optional[str] = None,
    ) -> InferenceEndpiont:
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
