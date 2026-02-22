import sqlite3

from vllm_omni.inputs.apertus_image_token_cache import (
    ApertusImageTokenCacheConfig,
    ApertusImageTokenSQLiteCache,
)


TABLE_NAME = "apertus_image_prompt_cache"


def _seed_disk_cache(db_path, cache_key, image_prompt):
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                cache_key TEXT PRIMARY KEY,
                image_prompt TEXT NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (cache_key, image_prompt)
            VALUES (?, ?)
            ON CONFLICT(cache_key) DO NOTHING
            """,
            (cache_key, image_prompt),
        )
        conn.commit()


def test_cache_config_defaults_are_5gib():
    config = ApertusImageTokenCacheConfig.from_mm_processor_kwargs({})
    assert config.ram_cache_kib == 5 * 1024 * 1024
    assert config.disk_cache_kib == 5 * 1024 * 1024
    assert config.disk_mmap_size_bytes == 5 * 1024 * 1024 * 1024
    assert config.preload_to_ram is True


def test_cache_config_mm_kwargs_override_defaults():
    config = ApertusImageTokenCacheConfig.from_mm_processor_kwargs(
        {
            "apertus_image_token_cache_preload_to_ram": False,
            "apertus_image_token_cache_ram_cache_kib": 1024,
            "apertus_image_token_cache_disk_cache_kib": 2048,
            "apertus_image_token_cache_disk_mmap_bytes": 4096,
            "apertus_image_token_cache_busy_timeout_ms": 7777,
            "apertus_image_token_cache_disk_synchronous": "full",
        }
    )
    assert config.preload_to_ram is False
    assert config.ram_cache_kib == 1024
    assert config.disk_cache_kib == 2048
    assert config.disk_mmap_size_bytes == 4096
    assert config.busy_timeout_ms == 7777
    assert config.disk_synchronous == "FULL"


def test_preload_to_ram_reads_existing_disk_even_if_disk_row_removed(tmp_path):
    db_path = tmp_path / "apertus_image_tokens.sqlite3"
    _seed_disk_cache(db_path, "key-a", "prompt-a")

    cache = ApertusImageTokenSQLiteCache(
        cache_db_path=db_path,
        table_name=TABLE_NAME,
        config=ApertusImageTokenCacheConfig.from_mm_processor_kwargs(
            {
                "apertus_image_token_cache_preload_to_ram": True,
                "apertus_image_token_cache_disk_mmap_bytes": 1024 * 1024,
            }
        ),
    )
    try:
        assert cache.get("key-a") == "prompt-a"
        # Delete on disk to verify RAM copy still serves it.
        cache._disk_conn.execute(f"DELETE FROM {TABLE_NAME} WHERE cache_key = ?", ("key-a",))
        cache._disk_conn.commit()
        assert cache.get("key-a") == "prompt-a"
    finally:
        cache.close()


def test_disk_fallback_promotes_to_ram_when_preload_disabled(tmp_path):
    db_path = tmp_path / "apertus_image_tokens.sqlite3"
    _seed_disk_cache(db_path, "key-b", "prompt-b")

    cache = ApertusImageTokenSQLiteCache(
        cache_db_path=db_path,
        table_name=TABLE_NAME,
        config=ApertusImageTokenCacheConfig.from_mm_processor_kwargs(
            {
                "apertus_image_token_cache_preload_to_ram": False,
                "apertus_image_token_cache_disk_mmap_bytes": 1024 * 1024,
            }
        ),
    )
    try:
        # First read is disk fallback + promotion.
        assert cache.get("key-b") == "prompt-b"
        cache._disk_conn.execute(f"DELETE FROM {TABLE_NAME} WHERE cache_key = ?", ("key-b",))
        cache._disk_conn.commit()
        # Second read is from promoted RAM copy.
        assert cache.get("key-b") == "prompt-b"
    finally:
        cache.close()


def test_put_writes_to_ram_and_flushes_to_disk(tmp_path):
    db_path = tmp_path / "apertus_image_tokens.sqlite3"
    cache = ApertusImageTokenSQLiteCache(
        cache_db_path=db_path,
        table_name=TABLE_NAME,
        config=ApertusImageTokenCacheConfig.from_mm_processor_kwargs(
            {
                "apertus_image_token_cache_preload_to_ram": False,
                "apertus_image_token_cache_disk_mmap_bytes": 1024 * 1024,
            }
        ),
    )
    try:
        cache.put("key-c", "prompt-c")
        assert cache.get("key-c") == "prompt-c"
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                f"SELECT image_prompt FROM {TABLE_NAME} WHERE cache_key = ?",
                ("key-c",),
            ).fetchone()
        assert row is not None
        assert row[0] == "prompt-c"
    finally:
        cache.close()


def test_disk_connection_uses_wal_and_read_pragmas(tmp_path):
    db_path = tmp_path / "apertus_image_tokens.sqlite3"
    config = ApertusImageTokenCacheConfig.from_mm_processor_kwargs(
        {
            "apertus_image_token_cache_preload_to_ram": False,
            "apertus_image_token_cache_disk_cache_kib": 4096,
            "apertus_image_token_cache_disk_mmap_bytes": 1024 * 1024,
        }
    )
    cache = ApertusImageTokenSQLiteCache(
        cache_db_path=db_path,
        table_name=TABLE_NAME,
        config=config,
    )
    try:
        journal_mode = cache._disk_conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        temp_store = int(cache._disk_conn.execute("PRAGMA temp_store").fetchone()[0])
        cache_size = int(cache._disk_conn.execute("PRAGMA cache_size").fetchone()[0])
        mmap_size = int(cache._disk_conn.execute("PRAGMA mmap_size").fetchone()[0])

        assert journal_mode == "wal"
        assert temp_store == 2  # MEMORY
        assert cache_size == -4096
        assert mmap_size >= 1024 * 1024
    finally:
        cache.close()
