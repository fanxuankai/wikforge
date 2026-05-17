"""Focused tests for ``IndexingService._upsert_qdrant`` (任务 12.5).

These tests harden the Qdrant-side write path: batching, payload contents,
vector layout, point-id validation, and error propagation.

The ``qdrant_client`` Python SDK isn't installed in every dev environment
(it's optional for running unit tests). To keep this file importable in any
venv, we install lightweight stand-ins for the few ``qdrant_client.*`` symbols
that ``app.core.qdrant`` and ``app.services.indexing_service`` import. The
stubs match the SDK's surface closely enough that the production code under
test can build ``PointStruct`` / ``SparseVector`` instances and we can assert
on their fields.
"""

from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ─── qdrant_client stubs ─────────────────────────────────────────────
#
# Installed BEFORE importing app.* so that ``from qdrant_client import ...``
# inside production code resolves to these dataclass placeholders rather than
# raising ModuleNotFoundError.


def _install_qdrant_stubs() -> None:
    # 如果真实 qdrant_client 已可用, 不要安装 stubs 以免污染其它测试。
    # 这个 stub 只是为了让没装 qdrant-client 的 dev 环境也能 collect 这个文件。
    try:
        import qdrant_client  # noqa: F401
        import qdrant_client.models  # noqa: F401
        import qdrant_client.http.exceptions  # noqa: F401

        return
    except ImportError:
        pass

    if "qdrant_client" in sys.modules and getattr(
        sys.modules["qdrant_client"], "_wikforge_test_stub", False
    ):
        return  # idempotent

    qdrant_pkg = types.ModuleType("qdrant_client")
    qdrant_pkg._wikforge_test_stub = True  # type: ignore[attr-defined]

    class _StubQdrantClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            self._args = args
            self._kwargs = kwargs

        def upsert(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return None

        def delete(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return None

        def get_collections(self) -> Any:  # pragma: no cover
            return MagicMock(collections=[])

        def create_collection(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return None

        def delete_collection(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return None

        def close(self) -> None:  # pragma: no cover
            return None

    qdrant_pkg.QdrantClient = _StubQdrantClient  # type: ignore[attr-defined]

    # qdrant_client.http.exceptions
    http_pkg = types.ModuleType("qdrant_client.http")
    exc_pkg = types.ModuleType("qdrant_client.http.exceptions")

    class _UnexpectedResponse(Exception):
        pass

    exc_pkg.UnexpectedResponse = _UnexpectedResponse  # type: ignore[attr-defined]
    http_pkg.exceptions = exc_pkg  # type: ignore[attr-defined]

    # qdrant_client.models
    models_pkg = types.ModuleType("qdrant_client.models")

    @dataclass
    class _PointStruct:
        id: Any = ""
        vector: Any = field(default_factory=dict)
        payload: dict = field(default_factory=dict)

    @dataclass
    class _SparseVector:
        indices: list = field(default_factory=list)
        values: list = field(default_factory=list)

    @dataclass
    class _Distance:
        COSINE: str = "Cosine"

    @dataclass
    class _VectorParams:
        size: int = 0
        distance: str = ""

    @dataclass
    class _SparseVectorParams:
        index: Any = None

    @dataclass
    class _SparseIndexParams:
        on_disk: bool = False

    @dataclass
    class _Filter:
        must: list = field(default_factory=list)

    @dataclass
    class _FieldCondition:
        key: str = ""
        match: Any = None

    @dataclass
    class _MatchValue:
        value: Any = None

    models_pkg.PointStruct = _PointStruct  # type: ignore[attr-defined]
    models_pkg.SparseVector = _SparseVector  # type: ignore[attr-defined]
    models_pkg.Distance = _Distance()  # type: ignore[attr-defined]
    models_pkg.VectorParams = _VectorParams  # type: ignore[attr-defined]
    models_pkg.SparseVectorParams = _SparseVectorParams  # type: ignore[attr-defined]
    models_pkg.SparseIndexParams = _SparseIndexParams  # type: ignore[attr-defined]
    models_pkg.Filter = _Filter  # type: ignore[attr-defined]
    models_pkg.FieldCondition = _FieldCondition  # type: ignore[attr-defined]
    models_pkg.MatchValue = _MatchValue  # type: ignore[attr-defined]

    sys.modules["qdrant_client"] = qdrant_pkg
    sys.modules["qdrant_client.http"] = http_pkg
    sys.modules["qdrant_client.http.exceptions"] = exc_pkg
    sys.modules["qdrant_client.models"] = models_pkg


_install_qdrant_stubs()


# Now safe to import production code.
from app.services.embedding_service import EmbeddingResult  # noqa: E402
from app.services.indexing_service import (  # noqa: E402
    QDRANT_BATCH_SIZE,
    ChunkPayload,
    IndexingError,
    IndexingService,
)
from app.core.qdrant import COLLECTION_NAME as QD_COLLECTION_NAME  # noqa: E402


DENSE_DIM = 1024


def _make_payload(chunk_id: str | None = None, **overrides: Any) -> ChunkPayload:
    base = dict(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        space_id=str(uuid.uuid4()),
        chunk_index=0,
        title_chain="H1 > H2 > H3",
        source_file="manual.pdf",
        page_number=1,
        content="hello world",
        parent_chunk_id=None,
        depth=2,
        token_count=12,
        allowed_user_ids=[str(uuid.uuid4())],
        access_level="read",
    )
    base.update(overrides)
    return ChunkPayload(**base)


def _make_embedding(chunk_id: str, *, sparse: bool = True) -> EmbeddingResult:
    return EmbeddingResult(
        chunk_id=chunk_id,
        dense_vector=[0.01] * DENSE_DIM,
        sparse_indices=[3, 17, 42] if sparse else [],
        sparse_values=[0.5, 0.7, 0.9] if sparse else [],
    )


def _make_service(qdrant_mock: MagicMock | None = None) -> IndexingService:
    service = IndexingService()
    service._qdrant = qdrant_mock if qdrant_mock is not None else MagicMock()
    # Stub OpenSearch path so we can isolate the Qdrant write under test —
    # _upsert_qdrant is exercised directly, but a few cases drive the full
    # index_chunks flow and we don't want OpenSearch noise.
    service._opensearch = MagicMock()
    return service


# ─── Empty / no-op cases ─────────────────────────────────────────────


def test_upsert_qdrant_empty_input_is_noop():
    """Empty payload list returns 0 and never calls the Qdrant client."""
    qd = MagicMock()
    service = _make_service(qd)

    indexed = service._upsert_qdrant([], [])

    assert indexed == 0
    qd.upsert.assert_not_called()


def test_index_chunks_empty_input_returns_zero_counts():
    service = _make_service()
    result = service.index_chunks([], [])
    assert result == {"qdrant_count": 0, "opensearch_count": 0}


# ─── PointStruct contents ───────────────────────────────────────────


def test_upsert_qdrant_uses_chunk_id_as_point_id_and_target_collection():
    qd = MagicMock()
    service = _make_service(qd)

    chunk_id = str(uuid.uuid4())
    payload = _make_payload(chunk_id=chunk_id)
    embedding = _make_embedding(chunk_id)

    indexed = service._upsert_qdrant([payload], [embedding])

    assert indexed == 1
    qd.upsert.assert_called_once()
    call_kwargs = qd.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == QD_COLLECTION_NAME
    points = call_kwargs["points"]
    assert len(points) == 1
    assert points[0].id == chunk_id


def test_upsert_qdrant_payload_contains_all_required_design_fields():
    """The payload must mirror the design.md Qdrant data model exactly."""
    qd = MagicMock()
    service = _make_service(qd)

    parent_id = str(uuid.uuid4())
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    payload = _make_payload(
        chunk_index=7,
        title_chain="第一章 > 1.1 > 1.1.2",
        source_file="handbook.pdf",
        page_number=42,
        content="full chunk text",
        parent_chunk_id=parent_id,
        depth=3,
        token_count=256,
        allowed_user_ids=[user_a, user_b],
        access_level="write",
    )
    embedding = _make_embedding(payload.chunk_id)

    service._upsert_qdrant([payload], [embedding])

    point = qd.upsert.call_args.kwargs["points"][0]
    assert point.payload == {
        "document_id": payload.document_id,
        "space_id": payload.space_id,
        "chunk_index": 7,
        "title_chain": "第一章 > 1.1 > 1.1.2",
        "source_file": "handbook.pdf",
        "page_number": 42,
        "content": "full chunk text",
        "parent_chunk_id": parent_id,
        "depth": 3,
        "token_count": 256,
        "allowed_user_ids": [user_a, user_b],
        "access_level": "write",
    }


def test_upsert_qdrant_vector_has_dense_1024_and_sparse_keys():
    from qdrant_client.models import SparseVector

    qd = MagicMock()
    service = _make_service(qd)
    payload = _make_payload()
    embedding = _make_embedding(payload.chunk_id, sparse=True)

    service._upsert_qdrant([payload], [embedding])

    point = qd.upsert.call_args.kwargs["points"][0]
    assert set(point.vector.keys()) == {"dense", "sparse"}
    assert len(point.vector["dense"]) == DENSE_DIM
    sparse = point.vector["sparse"]
    assert isinstance(sparse, SparseVector)
    assert sparse.indices == [3, 17, 42]
    assert sparse.values == [0.5, 0.7, 0.9]


def test_upsert_qdrant_omits_sparse_when_embedding_has_no_sparse_signal():
    """When the sparse vector is empty, only the dense key is sent."""
    qd = MagicMock()
    service = _make_service(qd)
    payload = _make_payload()
    embedding = _make_embedding(payload.chunk_id, sparse=False)

    service._upsert_qdrant([payload], [embedding])

    point = qd.upsert.call_args.kwargs["points"][0]
    assert list(point.vector.keys()) == ["dense"]


# ─── Batching ───────────────────────────────────────────────────────


def test_upsert_qdrant_splits_into_batches_of_qdrant_batch_size():
    """Inputs larger than QDRANT_BATCH_SIZE are split across multiple upsert calls."""
    qd = MagicMock()
    service = _make_service(qd)

    # Arrange 2.5x batches to verify both full and partial batches.
    n = QDRANT_BATCH_SIZE * 2 + 7
    payloads = [_make_payload() for _ in range(n)]
    embeddings = [_make_embedding(p.chunk_id) for p in payloads]

    indexed = service._upsert_qdrant(payloads, embeddings)

    assert indexed == n
    # Expected number of upsert calls: ceil(n / QDRANT_BATCH_SIZE)
    expected_calls = (n + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE
    assert qd.upsert.call_count == expected_calls

    # Verify each batch carries at most QDRANT_BATCH_SIZE points and the
    # IDs across all calls cover the full input set exactly once.
    seen_ids: list[str] = []
    for call in qd.upsert.call_args_list:
        points = call.kwargs["points"]
        assert 1 <= len(points) <= QDRANT_BATCH_SIZE
        seen_ids.extend(p.id for p in points)
    assert seen_ids == [p.chunk_id for p in payloads]


def test_upsert_qdrant_single_upsert_per_point_even_with_sparse():
    """Each point is upserted exactly once; we don't write dense then re-write
    with sparse (which would create a transient sparse-less revision)."""
    qd = MagicMock()
    service = _make_service(qd)
    payloads = [_make_payload() for _ in range(3)]
    embeddings = [_make_embedding(p.chunk_id, sparse=True) for p in payloads]

    service._upsert_qdrant(payloads, embeddings)

    # All 3 points fit in one batch — exactly one upsert call expected.
    assert qd.upsert.call_count == 1
    points = qd.upsert.call_args.kwargs["points"]
    assert len(points) == 3
    for point in points:
        assert "dense" in point.vector and "sparse" in point.vector


# ─── Validation & error propagation ─────────────────────────────────


def test_upsert_qdrant_rejects_non_uuid_chunk_id():
    """Qdrant requires UUID-string or unsigned-int point IDs; non-UUID
    strings must fail fast with a clear message."""
    service = _make_service()
    payload = _make_payload(chunk_id="not-a-uuid")
    embedding = _make_embedding(payload.chunk_id)

    with pytest.raises(IndexingError, match="not a valid UUID"):
        service._upsert_qdrant([payload], [embedding])


def test_upsert_qdrant_propagates_client_exception():
    """When the Qdrant client raises, the error propagates unchanged so
    ``index_chunks`` can wrap it and trigger rollback."""
    qd = MagicMock()
    qd.upsert.side_effect = RuntimeError("connection refused")
    service = _make_service(qd)
    payload = _make_payload()
    embedding = _make_embedding(payload.chunk_id)

    with pytest.raises(RuntimeError, match="connection refused"):
        service._upsert_qdrant([payload], [embedding])


def test_index_chunks_wraps_qdrant_runtime_error_into_indexing_error():
    """End-to-end: a Qdrant client failure surfaces as IndexingError."""
    qd = MagicMock()
    qd.upsert.side_effect = RuntimeError("boom")
    service = _make_service(qd)

    payload = _make_payload()
    embedding = _make_embedding(payload.chunk_id)

    with pytest.raises(IndexingError, match="Qdrant write failed"):
        service.index_chunks([payload], [embedding])


def test_index_chunks_returns_indexed_qdrant_count(monkeypatch):
    """``index_chunks`` returns the actual number of points upserted to Qdrant."""
    qd = MagicMock()
    service = _make_service(qd)

    n = 5
    payloads = [_make_payload() for _ in range(n)]
    embeddings = [_make_embedding(p.chunk_id) for p in payloads]

    # Stub the OpenSearch path to a no-op so we can assert on the count.
    monkeypatch.setattr(service, "_bulk_index_opensearch", lambda _payloads: None)

    result = service.index_chunks(payloads, embeddings)

    assert result["qdrant_count"] == n
    assert result["opensearch_count"] == n
