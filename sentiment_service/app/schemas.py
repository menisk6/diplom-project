from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RegisterModelRequest(BaseModel):
    family: str = Field(..., examples=['base-rubert'])
    source: str = Field(..., examples=['cointegrated/rubert-tiny2'])
    source_type: str = Field('hf', examples=['hf', 'local'])
    description: Optional[str] = None
    active: bool = True
    version: Optional[int] = None


class ModelRecord(BaseModel):
    model_id: str
    family: str
    version: int
    source: str
    source_type: str
    description: Optional[str] = None
    created_at: str
    active: bool
    labels: list[str] = []
    num_labels: int = 0


class FamilySummary(BaseModel):
    family: str
    active_version: Optional[int] = None
    versions: list[ModelRecord] = []


class PredictRequest(BaseModel):
    text: str
    family: Optional[str] = None
    version: Optional[int] = None
    model_id: Optional[str] = None
    top_k: int = 3


class PredictResponse(BaseModel):
    text: str
    model: ModelRecord
    prediction: dict[str, Any]
