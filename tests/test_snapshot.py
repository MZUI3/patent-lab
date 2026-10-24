# 색인 스냅샷은 만든 임베딩 모델과 현재 설정이 같을 때만 복원한다.
import json

import pytest

from src.vectorstore import snapshot


def test_legacy_snapshot_without_manifest_is_openai_small(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "MANIFEST", tmp_path / "manifest.json")
    assert snapshot.embedding_model() == "text-embedding-3-small"


def test_compatible_only_when_model_matches(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"embedding_provider": "local", "embedding_model": "BAAI/bge-m3", "dim": 1024}),
                        encoding="utf-8")
    monkeypatch.setattr(snapshot, "MANIFEST", manifest)
    monkeypatch.setattr(snapshot, "EMBEDDING_MODEL", "BAAI/bge-m3")
    assert snapshot.compatible()
    monkeypatch.setattr(snapshot, "EMBEDDING_MODEL", "text-embedding-3-small")
    assert not snapshot.compatible()


def test_restore_refuses_incompatible_snapshot(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"embedding_model": "BAAI/bge-m3"}), encoding="utf-8")
    monkeypatch.setattr(snapshot, "MANIFEST", manifest)
    monkeypatch.setattr(snapshot, "EMBEDDING_MODEL", "other-model")
    with pytest.raises(RuntimeError, match="복원하지 않습니다"):
        snapshot.restore()
