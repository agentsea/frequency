from __future__ import annotations
from typing import Optional, List
import os

from google.cloud import storage

from frequency.api.v1.server.models import V1Adapter as V1AdapterSchema
from frequency.db.conn import WithDB
from frequency.db.models import V1AdapterRecord
from .util import parse_gcs_uri


CACHE_DIR = os.getenv("ADAPTER_CACHE", "./.adapter")


class Adapter(WithDB):
    """An adapter for ML models"""

    name: str
    uri: str
    model: str

    def __init__(self, name: str, uri: str, model: str) -> None:
        self.name = name
        self.uri = uri
        self.model = model
        self.cache()
        self.save()

    def to_v1_schema(self) -> V1AdapterSchema:
        return V1AdapterSchema(name=self.name, uri=self.uri, model=self.model)

    @classmethod
    def from_v1_schema(cls, adapter: V1AdapterSchema) -> Adapter:
        return cls(name=adapter.name, uri=adapter.uri, model=adapter.model)

    def to_v1_record(self) -> V1AdapterRecord:
        return V1AdapterRecord(name=self.name, uri=self.uri, model=self.model)

    @classmethod
    def from_v1_record(cls, record: V1AdapterRecord) -> Adapter:
        return cls(name=record.name, uri=record.uri, model=record.model)

    def save(self) -> None:
        for db in self.get_db():
            record = self.to_v1_record()
            db.merge(record)
            db.commit()

    @classmethod
    def find(cls, name: str) -> Optional[Adapter]:
        for db in cls.get_db():
            record: Optional[V1AdapterRecord] = (
                db.query(V1AdapterRecord).filter_by(name=name).first()
            )
            if record:
                return cls.from_v1_record(record)

    @classmethod
    def list_v1(cls) -> List[V1AdapterSchema]:
        """
        List all adapters in V1 schema format.
        """
        adapters = []
        for db in cls.get_db():
            records = db.query(V1AdapterRecord).all()
            for record in records:
                adapter_instance = cls.from_v1_record(record)
                adapters.append(adapter_instance.to_v1_schema())
        return adapters

    def path(self) -> str:
        return os.path.join(CACHE_DIR, self.name)

    @classmethod
    def path_for_name(cls, name: str) -> str:
        return os.path.join(CACHE_DIR, name)

    def cache(self) -> str:
        """Cache the adapter to a local path"""

        if self.uri.startswith("gs://"):
            bucket_name, object_key = parse_gcs_uri(self.uri)
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(object_key)

            path = self.path()
            blob.download_to_filename(path)

            print(f"cached adapter '{self.uri}' to path {path}")
            return path

    @classmethod
    def delete(cls, name: str) -> bool:
        """
        Delete an adapter by name.
        Returns True if the adapter was successfully deleted, False otherwise.
        """
        for db in cls.get_db():
            record: Optional[V1AdapterRecord] = (
                db.query(V1AdapterRecord).filter_by(name=name).first()
            )
            if record:
                db.delete(record)
                db.commit()
                return True
            return False
