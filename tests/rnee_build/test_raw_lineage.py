import hashlib


def make_row_id(file_sha256: str, source_row_number: int) -> str:
    return hashlib.sha256(
        f"{file_sha256}:{source_row_number}".encode("ascii")
    ).hexdigest()


def test_row_id_is_deterministic_and_row_sensitive() -> None:
    source_hash = "a" * 64
    assert make_row_id(source_hash, 0) == make_row_id(source_hash, 0)
    assert make_row_id(source_hash, 0) != make_row_id(source_hash, 1)


def test_row_id_is_file_sensitive() -> None:
    assert make_row_id("a" * 64, 7) != make_row_id("b" * 64, 7)
