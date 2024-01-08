from sqlalchemy import Column, Integer, String, ForeignKey, Table, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB  # If using PostgreSQL

Base = declarative_base()

# If you're not using PostgreSQL, you might need to replace JSONB with a different type


class V1HealthRecord(Base):
    __tablename__ = "v1_health"
    id = Column(Integer, primary_key=True)
    status = Column(String)


class V1InfoRecord(Base):
    __tablename__ = "v1_info"
    id = Column(Integer, primary_key=True)
    version = Column(String)


class V1LoadModelRequestRecord(Base):
    __tablename__ = "v1_load_model_request"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    hf_repo = Column(String)


class V1ModelRecord(Base):
    __tablename__ = "v1_model"
    name = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=False)
    hf_repo = Column(String)
    cuda = Column(Boolean)
    # For adapters, assuming a many-to-many relationship
    adapters = relationship("V1AdapterRecord", secondary="model_adapter_link")


class V1AdapterRecord(Base):
    __tablename__ = "v1_adapter"
    name = Column(String, primary_key=True)
    model = Column(String, nullable=False)
    uri = Column(String)
    hf_repo = Column(String)


# Many-to-Many Link Table for V1ModelRecord and V1AdapterRecord
model_adapter_link = Table(
    "model_adapter_link",
    Base.metadata,
    Column("model_name", Integer, ForeignKey("v1_model.name")),
    Column("adapter_name", Integer, ForeignKey("v1_adapter.name")),
)


class V1ChatHistoryRecord(Base):
    __tablename__ = "v1_chat_history"
    id = Column(Integer, primary_key=True)
    history = Column(Text)


class V1ChatRequestRecord(Base):
    __tablename__ = "v1_chat_request"
    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    history_id = Column(Integer, ForeignKey("v1_chat_history.id"))


class V1ChatResponseRecord(Base):
    __tablename__ = "v1_chat_response"
    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    history_id = Column(Integer, ForeignKey("v1_chat_history.id"))
