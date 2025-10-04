import os
from pathlib import Path

from app.config import Settings


def test_settings_allowed_origins_parsing(monkeypatch):
	monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.com, https://b.com ,  http://c.com")
	s = Settings()
	assert s.allowed_origins == ["http://a.com", "https://b.com", "http://c.com"]


def test_settings_file_extensions_parsing(monkeypatch):
	monkeypatch.setenv("ALLOWED_FILE_EXTENSIONS", ".txt,.PDF , .Md")
	s = Settings()
	assert ".txt" in s.allowed_file_extensions
	assert ".pdf" in s.allowed_file_extensions
	assert ".md" in s.allowed_file_extensions


def test_settings_paths_and_ensure_directories(tmp_path: Path, monkeypatch):
	upload_dir = tmp_path / "docs"
	sqlite_path = tmp_path / "db" / "index.db"
	monkeypatch.setenv("UPLOAD_DIRECTORY", str(upload_dir))
	monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
	s = Settings()
	s.ensure_directories()
	assert Path(s.upload_directory).exists()
	assert Path(s.sqlite_path).parent.exists()


def test_settings_defaults_and_limits(monkeypatch):
	monkeypatch.delenv("ALLOWED_FILE_EXTENSIONS", raising=False)
	s = Settings()
	assert s.max_upload_mb >= 1
	assert isinstance(s.allowed_file_extensions, set)
