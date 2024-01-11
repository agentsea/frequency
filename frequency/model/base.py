from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftMixedModel
from accelerate import Accelerator

from frequency.api.v1.server.models import V1Model, V1GenerateResponse
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
    cuda: bool

    def __init__(
        self,
        name: str,
        type: str,
        hf_repo: str,
        adapters: List[str] = [],
        cuda: bool = True,
    ) -> None:
        self.name = name
        self.type = type
        self.hf_repo = hf_repo
        self.adapters = adapters
        self.cuda = cuda
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
        out.cuda = model.cuda

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
            cuda=self.cuda,
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
        out.cuda = record.cuda
        out.adapters = adapters
        return out

    def save(self) -> None:
        for db in self.get_db():
            print("saving: ", self.__dict__)
            record = self.to_v1_record()
            print("record: ", self.__dict__)
            db.merge(record)
            db.commit()

    @classmethod
    def find(cls, name: str) -> Optional[Model]:
        for db in cls.get_db():
            print("getting record..")
            record: Optional[V1ModelRecord] = (
                db.query(V1ModelRecord).filter_by(name=name).first()
            )
            print("model record: ", record)
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
            print(f"loading repo: {self.hf_repo}")
            if self.cuda:
                model = AutoModelForCausalLM.from_pretrained(
                    self.hf_repo, device_map="cuda", trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.hf_repo, trust_remote_code=True
                )
            # model = accelerator.prepare(model)
            # TODO: lock
            MODELS[self.name] = LoadedModel(model, tokenizer)

        else:
            raise ValueError(f"Model type unkown {self.type}")

    def get_class(self) -> Optional[LoadedModel]:
        return MODELS.get(self.name)

    def add_adapter(self, adapter: Adapter) -> None:
        loaded = self.get_class()
        if not loaded:
            raise ValueError("could not find model, was it loaded?")

        print(f"adding adapter name: '{adapter.name}' repo: '{adapter.hf_repo}' ...")
        loaded.model.load_adapter(adapter.hf_repo, adapter_name=adapter.name)
        print("added adapter")
        self.adapters.append(adapter)
        self.save()

    def delete_adapter(self, name: str) -> None:
        loaded = self.get_class()
        if not loaded:
            raise ValueError("could not find model, was it loaded?")

        print(f"deleting adapter {name}...")
        loaded.model.delete_adapter(name)
        print("delete adapter")

        adapters = []
        for adapter in self.adapters:
            if adapter.name == name:
                continue
            adapters.append(adapter)
        self.adapters = adapters
        self.save()

    def generate_v1(self, query: str, adapters: List[str] = []) -> V1GenerateResponse:
        loaded = self.get_class()
        print("loaded class")
        if adapters:
            print("using adapters: ", adapters)
            if len(adapters) > 1:
                raise ValueError(
                    "multiple adapters not yet supported https://github.com/huggingface/transformers/issues/28372"
                )
            print(f"setting adapter: {adapters[0]}")
            loaded.model.set_adapter(adapters[0])
        else:
            print("not using any adapters")

        print("generating")
        inputs = loaded.tokenizer(query, return_tensors="pt")
        output = loaded.model.generate(**inputs)
        decoded_output = loaded.tokenizer.decode(output[0], skip_special_tokens=True)
        print("decoded output: ", decoded_output)
        return V1GenerateResponse(text=decoded_output)
