import os
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value_str = value.strip().lower()
        if value_str in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if value_str in {"0", "false", "f", "no", "n", "off"}:
            return False
    return default


def _coerce_int(value: Any, *, default: int, min_value: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _resolve_value(
    mm_processor_kwargs: Mapping[str, Any] | None,
    kwarg_name: str,
    env_var_name: str,
) -> Any:
    if mm_processor_kwargs is not None and kwarg_name in mm_processor_kwargs:
        return mm_processor_kwargs.get(kwarg_name)
    return os.getenv(env_var_name)


@dataclass(frozen=True)
class ApertusImageTokenCacheConfig:
    preload_to_ram: bool
    ram_cache_kib: int
    disk_cache_kib: int
    disk_mmap_size_bytes: int
    busy_timeout_ms: int
    disk_synchronous: str

    _DEFAULT_CACHE_BYTES = 5 * 1024 * 1024 * 1024
    _DEFAULT_CACHE_KIB = _DEFAULT_CACHE_BYTES // 1024

    _PRELOAD_ENV = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD_TO_RAM"
    _RAM_CACHE_KIB_ENV = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_RAM_CACHE_KIB"
    _DISK_CACHE_KIB_ENV = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DISK_CACHE_KIB"
    _DISK_MMAP_ENV = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DISK_MMAP_BYTES"
    _BUSY_TIMEOUT_ENV = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_BUSY_TIMEOUT_MS"
    _DISK_SYNC_ENV = "VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DISK_SYNCHRONOUS"

    @classmethod
    def from_mm_processor_kwargs(
        cls,
        mm_processor_kwargs: Mapping[str, Any] | None,
    ) -> "ApertusImageTokenCacheConfig":
        preload_to_ram = _coerce_bool(
            _resolve_value(
                mm_processor_kwargs,
                "apertus_image_token_cache_preload_to_ram",
                cls._PRELOAD_ENV,
            ),
            default=True,
        )
        ram_cache_kib = _coerce_int(
            _resolve_value(
                mm_processor_kwargs,
                "apertus_image_token_cache_ram_cache_kib",
                cls._RAM_CACHE_KIB_ENV,
            ),
            default=cls._DEFAULT_CACHE_KIB,
            min_value=1,
        )
        disk_cache_kib = _coerce_int(
            _resolve_value(
                mm_processor_kwargs,
                "apertus_image_token_cache_disk_cache_kib",
                cls._DISK_CACHE_KIB_ENV,
            ),
            default=cls._DEFAULT_CACHE_KIB,
            min_value=1,
        )
        disk_mmap_size_bytes = _coerce_int(
            _resolve_value(
                mm_processor_kwargs,
                "apertus_image_token_cache_disk_mmap_bytes",
                cls._DISK_MMAP_ENV,
            ),
            default=cls._DEFAULT_CACHE_BYTES,
            min_value=0,
        )
        busy_timeout_ms = _coerce_int(
            _resolve_value(
                mm_processor_kwargs,
                "apertus_image_token_cache_busy_timeout_ms",
                cls._BUSY_TIMEOUT_ENV,
            ),
            default=5000,
            min_value=1,
        )
        disk_synchronous = str(
            _resolve_value(
                mm_processor_kwargs,
                "apertus_image_token_cache_disk_synchronous",
                cls._DISK_SYNC_ENV,
            )
            or "NORMAL"
        ).upper()
        if disk_synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            disk_synchronous = "NORMAL"

        return cls(
            preload_to_ram=preload_to_ram,
            ram_cache_kib=ram_cache_kib,
            disk_cache_kib=disk_cache_kib,
            disk_mmap_size_bytes=disk_mmap_size_bytes,
            busy_timeout_ms=busy_timeout_ms,
            disk_synchronous=disk_synchronous,
        )


class ApertusImageTokenSQLiteCache:
    def __init__(
        self,
        cache_db_path: Path,
        *,
        table_name: str,
        config: ApertusImageTokenCacheConfig,
    ) -> None:
        self._cache_db_path = cache_db_path
        self._table_name = table_name
        self._config = config
        self._lock = threading.Lock()
        self._ram_conn: sqlite3.Connection | None = None
        self._disk_conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_s = max(self._config.busy_timeout_ms / 1000.0, 1.0)

        disk_conn = sqlite3.connect(
            str(self._cache_db_path),
            timeout=timeout_s,
            check_same_thread=False,
        )
        self._configure_disk_connection(disk_conn)
        self._create_table(disk_conn)
        disk_conn.commit()

        ram_conn = sqlite3.connect(
            ":memory:",
            timeout=timeout_s,
            check_same_thread=False,
        )
        self._configure_ram_connection(ram_conn)
        if self._config.preload_to_ram:
            disk_conn.backup(ram_conn)
        else:
            self._create_table(ram_conn)
            ram_conn.commit()

        self._disk_conn = disk_conn
        self._ram_conn = ram_conn

    def _configure_disk_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(f"PRAGMA synchronous={self._config.disk_synchronous};")
        conn.execute(f"PRAGMA busy_timeout={self._config.busy_timeout_ms};")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute(f"PRAGMA cache_size={-self._config.disk_cache_kib};")
        conn.execute(f"PRAGMA mmap_size={self._config.disk_mmap_size_bytes};")

    def _configure_ram_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=MEMORY;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute(f"PRAGMA busy_timeout={self._config.busy_timeout_ms};")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute(f"PRAGMA cache_size={-self._config.ram_cache_kib};")

    def _create_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                cache_key TEXT PRIMARY KEY,
                image_prompt TEXT NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
            )
            """
        )

    def _select_prompt(
        self,
        conn: sqlite3.Connection,
        cache_key: str,
    ) -> str | None:
        row = conn.execute(
            f"""
            SELECT image_prompt
            FROM {self._table_name}
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        prompt = row[0]
        return prompt if isinstance(prompt, str) and prompt else None

    def _insert_prompt(
        self,
        conn: sqlite3.Connection,
        cache_key: str,
        image_prompt: str,
    ) -> None:
        conn.execute(
            f"""
            INSERT INTO {self._table_name} (cache_key, image_prompt)
            VALUES (?, ?)
            ON CONFLICT(cache_key) DO NOTHING
            """,
            (cache_key, image_prompt),
        )

    def get(self, cache_key: str) -> str | None:
        ram_conn = self._ram_conn
        disk_conn = self._disk_conn
        if ram_conn is None or disk_conn is None:
            return None

        with self._lock:
            try:
                if ram_prompt := self._select_prompt(ram_conn, cache_key):
                    return ram_prompt

                disk_prompt = self._select_prompt(disk_conn, cache_key)
                if disk_prompt is None:
                    return None

                self._insert_prompt(ram_conn, cache_key, disk_prompt)
                ram_conn.commit()
                return disk_prompt
            except sqlite3.Error as exc:
                logger.warning(
                    "Failed reading Apertus image token SQLite cache %s: %s",
                    self._cache_db_path,
                    exc,
                )
                return None

    def put(self, cache_key: str, image_prompt: str) -> None:
        ram_conn = self._ram_conn
        disk_conn = self._disk_conn
        if ram_conn is None or disk_conn is None:
            return

        with self._lock:
            try:
                self._insert_prompt(ram_conn, cache_key, image_prompt)
                ram_conn.commit()
                self._insert_prompt(disk_conn, cache_key, image_prompt)
                disk_conn.commit()
            except sqlite3.Error as exc:
                logger.warning(
                    "Failed writing Apertus image token SQLite cache %s: %s",
                    self._cache_db_path,
                    exc,
                )
                try:
                    ram_conn.rollback()
                except sqlite3.Error:
                    pass
                try:
                    disk_conn.rollback()
                except sqlite3.Error:
                    pass

    def close(self) -> None:
        with self._lock:
            if self._ram_conn is not None:
                try:
                    self._ram_conn.close()
                except sqlite3.Error:
                    pass
                self._ram_conn = None
            if self._disk_conn is not None:
                try:
                    self._disk_conn.close()
                except sqlite3.Error:
                    pass
                self._disk_conn = None

