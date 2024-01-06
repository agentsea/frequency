from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftMixedModel
from accelerate import Accelerator

from frequency.api.v1.server.models import V1Model, V1ChatHistory, V1ChatResponse
from frequency.db.conn import WithDB
from frequency.db.models import V1ModelRecord
from frequency.adapter.base import Adapter

MODELS: Dict[str, LoadedModel] = {}

accelerator = Accelerator()


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


class Model(WithDB):
    """Am ML model reference"""

    name: str
    type: str
    hf_repo: str
    adapters: Optional[List[Adapter]] = None

    def __init__(
        self, name: str, type: str, hf_repo: str, adapters: Optional[List[str]] = None
    ) -> None:
        self.name = name
        self.type = type
        self.hf_repo = hf_repo
        self.adapters = adapters
        self.load()
        self.save()

    def to_v1_schema(self) -> V1Model:
        adapters = []
        for adapter in self.adapters:
            adapters.append(adapter.name)

        return V1Model(
            name=self.name, type=self.type, hf_repo=self.hf_repo, adapters=adapters
        )

    @classmethod
    def from_v1_schema(cls, model: V1Model) -> Model:
        out = cls.__new__(Model)
        out.name = model.name
        out.type = model.type
        out.hf_repo = model.hf_repo

        if model.adapters:
            adapters = []
            for adapter in model.adapters:
                adapters.append(Adapter.find(adapter))
            out.adapters = model.adapters
        return out

    def to_v1_record(self) -> V1ModelRecord:
        adapters = []
        for adapter in self.adapters:
            adapters.append(adapter.to_v1_record())

        return V1ModelRecord(
            name=self.name,
            type=self.type,
            hf_repo=self.hf_repo,
            adapters=adapters,
        )

    @classmethod
    def from_v1_record(cls, record: V1ModelRecord) -> Model:
        adapters = []
        for adapter in record.adapters:
            adapters.append(Adapter.from_v1_record(adapter))
        out = cls.__new__(Model)
        out.name = record.name
        out.type = record.type
        out.hf_repo = record.hf_repo
        out.adapters = adapters

    def save(self) -> None:
        for db in self.get_db():
            record = self.to_v1_record()
            db.merge(record)
            db.commit()

    @classmethod
    def find(cls, name: str) -> Optional[Model]:
        for db in cls.get_db():
            record: Optional[V1ModelRecord] = (
                db.query(V1ModelRecord).filter_by(name=name).first()
            )
            if record:
                return cls.from_v1_record(record)

    @classmethod
    def list_v1(cls) -> List[V1Model]:
        """
        List all models in V1 schema format.
        """
        models = []
        for db in cls.get_db():
            records = db.query(V1ModelRecord).all()
            for record in records:
                model_instance = cls.from_v1_record(record)
                models.append(model_instance.to_v1_schema())
        return models

    @classmethod
    def delete(cls, name: str) -> bool:
        """
        Delete a model by name.
        """
        for db in cls.get_db():
            record: Optional[V1ModelRecord] = (
                db.query(V1ModelRecord).filter_by(name=name).first()
            )
            if record:
                db.delete(record)
                db.commit()
                return True
            return False

    def load(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.hf_repo, trust_remote_code=True)

        if self.type == "AutoModelForCausalLM":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True
            ).eval()
            model = accelerator.prepare(model)
            # TODO: lock
            MODELS[self.name] = LoadedModel(model, tokenizer)

        else:
            raise ValueError(f"Model type unkown {self.type}")

    def get_class(self) -> Optional[LoadedModel]:
        return MODELS.get(self.name)

    def chat_v1(
        self,
        query: str,
        history: Optional[V1ChatHistory] = None,
        adapters: List[str] = [],
    ) -> V1ChatResponse:
        loaded = self.get_class()
        print("loaded class")
        if adapters:
            print("using adapters: ", adapters)
            adapter0 = Adapter.find(adapters[0])
            peft_model = PeftMixedModel.from_pretrained(
                loaded.model, adapter0.path(), adapter0.name
            )

            for name in adapters[1:]:
                adapter = Adapter.find(name)
                peft_model.load_adapter(adapter.path(), adapter_name=adapter.name)

            peft_model.set_adapter(adapters)

            inputs = loaded.tokenizer(query, return_tensors="pt")
            peft_model(**inputs)
        response, history = loaded.model.chat(
            loaded.tokenizer, query=query, history=history
        )
        print("\nresponse: ", response)
        print("\nhistory: ", history)

        return V1ChatResponse(text=response, history=V1ChatHistory(history))
