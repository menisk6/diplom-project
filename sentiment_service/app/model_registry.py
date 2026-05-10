from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .config import REGISTRY_PATH
from .preprocessing import preprocess_text


@dataclass
class ModelRecord:
    model_id: str
    family: str
    version: int
    source: str
    source_type: str  # hf | local
    description: Optional[str]
    created_at: str
    active: bool
    labels: list[str]
    num_labels: int


class ModelRegistry:
    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path
        self._models: list[ModelRecord] = []
        self._cache: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self._models = []
            return
        data = json.loads(self.registry_path.read_text(encoding='utf-8'))
        self._models = [ModelRecord(**item) for item in data.get('models', [])]

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'models': [asdict(m) for m in self._models]}
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _infer_labels(config) -> tuple[list[str], int]:
        labels = []
        if getattr(config, 'id2label', None):
            try:
                labels = [config.id2label[i] for i in range(len(config.id2label))]
            except Exception:
                try:
                    labels = [v for _, v in sorted(config.id2label.items(), key=lambda x: int(x[0]))]
                except Exception:
                    labels = []
        if not labels:
            num_labels = int(getattr(config, 'num_labels', 2) or 2)
            labels = [f'LABEL_{i}' for i in range(num_labels)]
        return labels, len(labels)

    def list_models(self) -> list[ModelRecord]:
        return list(self._models)

    def list_families(self) -> list[dict[str, Any]]:
        families = {}
        for m in self._models:
            families.setdefault(m.family, []).append(m)
        result = []
        for family, models in sorted(families.items()):
            active_version = None
            for m in models:
                if m.active:
                    active_version = m.version
                    break
            result.append({
                'family': family,
                'active_version': active_version,
                'versions': sorted(models, key=lambda x: x.version),
            })
        return result

    def get_model(self, model_id: str) -> ModelRecord:
        for m in self._models:
            if m.model_id == model_id:
                return m
        raise KeyError(f'Model id not found: {model_id}')

    def get_model_by_family_version(self, family: str, version: int) -> ModelRecord:
        for m in self._models:
            if m.family == family and m.version == version:
                return m
        raise KeyError(f'Model not found: {family} v{version}')

    def get_active_model(self, family: str) -> ModelRecord:
        candidates = [m for m in self._models if m.family == family and m.active]
        if candidates:
            return sorted(candidates, key=lambda x: x.version)[-1]
        candidates = [m for m in self._models if m.family == family]
        if not candidates:
            raise KeyError(f'No models for family: {family}')
        return sorted(candidates, key=lambda x: x.version)[-1]

    def _next_version(self, family: str) -> int:
        versions = [m.version for m in self._models if m.family == family]
        return (max(versions) + 1) if versions else 1

    def register_model(
        self,
        family: str,
        source: str,
        source_type: str = 'hf',
        description: Optional[str] = None,
        active: bool = True,
        version: Optional[int] = None,
    ) -> ModelRecord:
        if source_type not in {'hf', 'local'}:
            raise ValueError("source_type must be 'hf' or 'local'")

        if version is None:
            version = self._next_version(family)
        else:
            if any(m.family == family and m.version == version for m in self._models):
                raise ValueError(f'Version already exists: {family} v{version}')

        config = AutoConfig.from_pretrained(source)
        labels, num_labels = self._infer_labels(config)

        model = ModelRecord(
            model_id=str(uuid.uuid4()),
            family=family,
            version=version,
            source=source,
            source_type=source_type,
            description=description,
            created_at=self._now(),
            active=active,
            labels=labels,
            num_labels=num_labels,
        )

        if active:
            self._deactivate_family(family)

        self._models.append(model)
        self._save()
        return model

    def _deactivate_family(self, family: str) -> None:
        for m in self._models:
            if m.family == family:
                m.active = False

    def set_active(self, family: str, version: int) -> ModelRecord:
        target = self.get_model_by_family_version(family, version)
        self._deactivate_family(family)
        target.active = True
        self._save()
        self._cache.pop(target.model_id, None)
        return target

    def delete_version(self, family: str, version: int) -> None:
        target = self.get_model_by_family_version(family, version)
        self._models = [m for m in self._models if not (m.family == family and m.version == version)]
        self._cache.pop(target.model_id, None)

        remaining = [m for m in self._models if m.family == family]
        if remaining and not any(m.active for m in remaining):
            newest = sorted(remaining, key=lambda x: x.version)[-1]
            newest.active = True

        self._save()

    def _load_pipeline(self, record: ModelRecord) -> dict[str, Any]:
        if record.model_id in self._cache:
            return self._cache[record.model_id]

        source = record.source
        if record.source_type == 'local' and not Path(source).exists():
            raise FileNotFoundError(f'Local model path does not exist: {source}')

        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(source)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        labels = list(record.labels)
        if not labels:
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

        payload = {'tokenizer': tokenizer, 'model': model, 'device': device, 'labels': labels}
        self._cache[record.model_id] = payload
        return payload

    def predict(
        self,
        text: str,
        model_id: Optional[str] = None,
        family: Optional[str] = None,
        version: Optional[int] = None,
        top_k: int = 3,
        max_length: int = 256,
    ) -> dict[str, Any]:
        if model_id:
            record = self.get_model(model_id)
        elif family is not None and version is not None:
            record = self.get_model_by_family_version(family, version)
        elif family is not None:
            record = self.get_active_model(family)
        else:
            active = [m for m in self._models if m.active]
            if not active:
                raise ValueError('No active models registered')
            record = sorted(active, key=lambda x: (x.family, x.version))[-1]

        runtime = self._load_pipeline(record)
        tokenizer = runtime['tokenizer']
        model = runtime['model']
        device = runtime['device']
        labels = runtime['labels']

        cleaned = preprocess_text(text)
        inputs = tokenizer(
            cleaned,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()

        top_k = max(1, min(top_k, len(labels)))
        top_probs, top_idx = torch.topk(probs, k=top_k)

        scores = [
            {'label': labels[i], 'score': float(p)}
            for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]

        best = scores[0]
        return {
            'model': asdict(record),
            'input': cleaned,
            'prediction': best,
            'top_k': scores,
        }

    def export_model(self, model_id: str, destination: str) -> str:
        record = self.get_model(model_id)
        runtime = self._load_pipeline(record)
        model = runtime['model']
        tokenizer = runtime['tokenizer']

        dst = Path(destination)
        dst.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dst)
        tokenizer.save_pretrained(dst)

        meta = {
            'model': asdict(record),
            'exported_at': self._now(),
        }
        (dst / 'registry_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        return str(dst.resolve())

    def import_local_snapshot(self, family: str, source_dir: str, description: Optional[str] = None, active: bool = True) -> ModelRecord:
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(source_dir)
        return self.register_model(
            family=family,
            source=str(source.resolve()),
            source_type='local',
            description=description,
            active=active,
        )
