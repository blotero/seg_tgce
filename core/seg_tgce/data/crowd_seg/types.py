from typing import TypedDict


class InvertedMetadataRecord(TypedDict):
    total: int
    scored: list[str]
