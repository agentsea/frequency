from typing import Tuple, List

from .v1.frequency_api import FrequencyAPI
from frequency.api.v1.server.models import (
    V1ChatRequest,
    V1ChatResponse,
    V1LoadModelRequest,
    V1Adapter,
)


class ModelClient:
    """A client for a server model"""

    def __init__(self, addr: str, model_name: str) -> None:
        self._client = FrequencyAPI(endpoint=addr)
        self._model_name = model_name

    def chat(
        self, query: str, history: List, adapters: List[str] = []
    ) -> Tuple[str, List]:
        """Chat with the model.

        Args:
            query (str): Query to chat with the model.
            history (List): Chat history.
            adapters (List[str], optional): List of adapters to add to the call. Defaults to [].

        Returns:
            Tuple[str, List]: Response and history
        """
        req = V1ChatRequest(query=query, history=history, adapters=adapters)
        resp = self._client.chat(self._model_name, req.__dict__)
        chat_resp = V1ChatResponse(**resp)

        return chat_resp.text, chat_resp.history

    def load_adapter(self, hf_repo: str, adapter_name: str) -> None:
        """Load the adapter.

        Args:
            hf_repo (str): HF repo to load the adapter from.
            adapter_name (str): Name the adapter.
        """
        adapter = V1Adapter(name=adapter_name, hf_repo=hf_repo, model=self._model_name)
        self._client.load_adapter(adapter.__dict__)
        return


class FrequencyClient:
    """A client for the frequency server"""

    def __init__(self, addr: str) -> None:
        self._addr = addr
        self._client = FrequencyAPI(endpoint=addr)

    def load_model(
        self,
        hf_repo: str,
        name: str,
        type: str = "AutoModelForCausalLM",
        cuda: bool = True,
    ) -> ModelClient:
        """Load a model.

        Args:
            hf_repo (str): HF repo to load the model from.
            name (str): Name the model.
            type (str, optional): HF type. Defaults to "AutoModelForCausalLM".
            cuda (bool, optional): Whether to use cuda. Defaults to True.
        """
        req = V1LoadModelRequest(hf_repo=hf_repo, name=name, type=type, cuda=cuda)
        self._client.load_model(req.__dict__)
        return ModelClient(addr=self._addr, model_name=name)

    def chat(
        self, model_name: str, query: str, history: List, adapters: List[str] = []
    ) -> Tuple[str, List]:
        """Chat with the model.

        Args:
            model_name (str): Name of the model to chat with.
            query (str): Query to chat with the model.
            history (List): Chat history.
            adapters (List[str], optional): List of adapters to add to the call. Defaults to [].

        Returns:
            Tuple[str, List]: Response and history
        """
        req = V1ChatRequest(query=query, history=history, adapters=adapters)
        resp = self._client.chat(model_name, req.__dict__)
        chat_resp = V1ChatResponse(**resp)

        return chat_resp.text, chat_resp.history

    def load_adapter(self, model_name: str, hf_repo: str, adapter_name: str) -> None:
        """Load an adapter for a model.

        Args:
            model_name (str): Model name to use
            hf_repo (str): HF repo of the adapter
            adapter_name (str): Name the adapter
        """
        adapter = V1Adapter(name=adapter_name, hf_repo=hf_repo, model=model_name)
        self._client.load_adapter(adapter.__dict__)
        return
