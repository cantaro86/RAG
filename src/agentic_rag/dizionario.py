import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import pandas as pd

from agentic_rag.loggers import Logger

logger = Logger.get_logger(__name__)

_SPLIT_RE = re.compile(r"\s*[/,]\s*")
_SPACE_RE = re.compile(r"\s+")


def normalize(term: str) -> str:
    return _SPACE_RE.sub(" ", str(term).strip().lower())


def split_terms(text) -> list[str]:
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]


def dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = normalize(item)
        if key and key not in seen:
            seen.add(key)
            out.append(str(item).strip())
    return out


def merge_overlapping_groups(groups: list[list[str]]) -> list[list[str]]:
    pending = [dedup_preserve_order(group) for group in groups if group]
    merged: list[list[str]] = []

    while pending:
        base = pending.pop(0)
        base_keys = {normalize(x) for x in base}

        changed = True
        while changed:
            changed = False
            rest = []
            for other in pending:
                other_keys = {normalize(x) for x in other}
                if base_keys & other_keys:
                    base = dedup_preserve_order([*base, *other])
                    base_keys = {normalize(x) for x in base}
                    changed = True
                else:
                    rest.append(other)
            pending = rest

        merged.append(base)

    return merged


def load_synonym_groups_from_excel(path: str | Path) -> tuple[tuple[str, ...], ...]:
    logger.debug(f"Loading synonym groups from Excel: {path}")
    df = pd.read_excel(Path(path), header=1, names=["term", "synonyms"])

    raw_groups = []
    for _, row in df.iterrows():
        group = dedup_preserve_order([*split_terms(row["term"]), *split_terms(row["synonyms"])])
        if group:
            raw_groups.append(group)

    merged = merge_overlapping_groups(raw_groups)
    logger.debug(f"Loaded {len(merged)} synonym group(s) from {path}")
    return tuple(tuple(group) for group in merged)


@dataclass(frozen=True)
class GroupMatcher:
    group: tuple[str, ...]
    normalized_terms: tuple[str, ...]
    patterns: tuple[re.Pattern, ...]


class SynonymStore:
    def __init__(self, excel_path: str | Path):
        self.path = Path(excel_path)
        self._lock = RLock()
        self._mtime: float | None = None
        self._groups: tuple[tuple[str, ...], ...] = ()
        self._matchers: tuple[GroupMatcher, ...] = ()
        self.reload_if_needed(force=True)

    def _build_matchers(
        self,
        groups: tuple[tuple[str, ...], ...],
    ) -> tuple[GroupMatcher, ...]:
        matchers = []
        for group in groups:
            normalized_pairs = sorted(
                ((term, normalize(term)) for term in group),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            original_terms = tuple(term for term, _ in normalized_pairs)
            normalized_terms = tuple(term_norm for _, term_norm in normalized_pairs)
            patterns = tuple(
                re.compile(r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)") for term_norm in normalized_terms
            )
            matchers.append(
                GroupMatcher(
                    group=original_terms,
                    normalized_terms=normalized_terms,
                    patterns=patterns,
                )
            )
        return tuple(matchers)

    def reload_if_needed(self, force: bool = False) -> None:
        with self._lock:
            mtime = self.path.stat().st_mtime
            if not force and self._mtime == mtime:
                return

            groups = load_synonym_groups_from_excel(self.path)
            self._groups = groups
            self._matchers = self._build_matchers(groups)
            self._mtime = mtime
            logger.info(f"SynonymStore reloaded from {self.path} with {len(groups)} group(s)")

    def _get_groups_ref(self) -> tuple[tuple[str, ...], ...]:
        self.reload_if_needed()
        return self._groups

    def get_groups(self) -> list[list[str]]:
        self.reload_if_needed()
        return [list(group) for group in self._groups]

    def _find_matches(
        self,
        text: str,
    ) -> list[tuple[str, tuple[str, ...]]]:
        self.reload_if_needed()
        text_norm = normalize(text)

        matches = []
        for matcher in self._matchers:
            found_term = None
            for original, pattern in zip(matcher.group, matcher.patterns, strict=False):
                if pattern.search(text_norm):
                    found_term = original
                    break

            if found_term is not None:
                matches.append((found_term, matcher.group))

        return matches

    def find_matching_groups(self, text: str) -> list[list[str]]:
        matches = self._find_matches(text)
        result = [list(group) for _, group in matches]
        logger.info(f"find_matching_groups returned {len(result)} group(s) for text: {text}")
        logger.debug(f"find_matching_groups detail: {result}")
        return result

    def find_matched_terms(self, text: str) -> dict[str, list[str]]:
        matches = self._find_matches(text)
        result = {
            found_term: [term for term in group if normalize(term) != normalize(found_term)]
            for found_term, group in matches
        }
        logger.info(f"find_matched_terms returned {len(result)} match(es) for text: {text}")
        logger.debug(f"find_matched_terms detail: {result}")
        return result

    def collect_expansion_terms(self, text: str) -> list[str]:
        matches = self._find_matches(text)
        return dedup_preserve_order(term for _, group in matches for term in group)
