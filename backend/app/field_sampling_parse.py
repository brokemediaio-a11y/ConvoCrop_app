"""
Parse free-text field sampling messages into FieldSamplingReport.

Users often paste counts in chat instead of JSON; this extracts pairs of
(total plants, infected plants), field area, and optional quadrat radius.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.schemas import AreaUnit, FieldSample, FieldSamplingReport

_TOTAL_RE = re.compile(
    r"total\s*plants?\s*[:.]?\s*(\d+)",
    re.IGNORECASE,
)
_INFECTED_RE = re.compile(
    r"infected\s*plants?\s*[:.]?\s*(\d+)",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"part\s*(\d)", re.IGNORECASE)


def _extract_totals_infected(text: str) -> Tuple[List[int], List[int]]:
    totals = [int(x) for x in _TOTAL_RE.findall(text)]
    infected = [int(x) for x in _INFECTED_RE.findall(text)]
    return totals, infected


def looks_like_field_sample_submission(text: str) -> bool:
    """User is likely pasting quadrat / sample counts."""
    totals, infected = _extract_totals_infected(text)
    if len(totals) >= 2 and len(totals) == len(infected):
        return True
    if len(totals) == 1 and len(infected) == 1:
        tl = text.lower()
        return any(
            k in tl
            for k in (
                "total area",
                "field area",
                "meter square",
                "square meter",
                "hectare",
                "acre",
                "kanal",
                "marla",
                "quadrat",
                "radius",
            )
        )
    return False


def _parse_field_area(text: str) -> Optional[Tuple[float, AreaUnit]]:
    tl = text.lower()

    m = re.search(
        r"(?:total\s*)?(?:field\s*)?area\s*[:.]?\s*([\d.]+)\s*kanals?\b",
        tl,
    )
    if m:
        return float(m.group(1)), "kanal"

    m = re.search(
        r"(?:total\s*)?(?:field\s*)?area\s*[:.]?\s*([\d.]+)\s*marlas?\b",
        tl,
    )
    if m:
        return float(m.group(1)), "marla"

    m = re.search(
        r"(?:total\s*)?(?:field\s*)?area\s*[:.]?\s*([\d.]+)\s*(ha|hectares?|acres?)\b",
        tl,
    )
    if m:
        val = float(m.group(1))
        return (val, "ha") if m.group(2).startswith("h") else (val, "acre")

    m = re.search(
        r"(?:total\s*)?(?:field\s*)?area\s*[:.]?\s*([\d.]+)\s*"
        r"(?:m\s*2|m2|m\^2|sq\.?\s*m|square\s*meters?|square\s*metres?|meter\s*square|metre\s*square)",
        tl,
    )
    if m:
        return float(m.group(1)), "m2"

    m = re.search(
        r"(?:total\s*)?(?:field\s*)?area\s*[:.]?\s*([\d.]+)\s*(?:meters?|metres?)\s*square",
        tl,
    )
    if m:
        return float(m.group(1)), "m2"

    return None


def _parse_quadrat_radius_m(text: str) -> Optional[float]:
    tl = text.lower()
    patterns = [
        r"(?:quadrat|plot|frame|circle|circular)[^\n]*?(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*radius",
        r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*radius",
        r"radius\s*[:.]?\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?",
    ]
    for pat in patterns:
        m = re.search(pat, tl)
        if m:
            return float(m.group(1))
    return None


def _assign_quadrants(text: str, n: int) -> List[Optional[int]]:
    """Map each nth 'total plants' line to the latest 'part N' seen above it."""
    lines = text.splitlines()
    quads: List[Optional[int]] = []
    current_part: Optional[int] = None
    for line in lines:
        mp = _PART_RE.search(line)
        if mp:
            q = int(mp.group(1))
            if 1 <= q <= 4:
                current_part = q
        if _TOTAL_RE.search(line) and len(quads) < n:
            quads.append(current_part)
    while len(quads) < n:
        quads.append(None)
    return quads[:n]


def try_parse_field_sampling_from_text(text: str) -> Optional[FieldSamplingReport]:
    """
    If the message contains paired total/infected counts and a parseable field area,
    build a FieldSamplingReport.
    """
    if not text or not text.strip():
        return None

    totals, infected = _extract_totals_infected(text)
    if not totals or len(totals) != len(infected):
        return None
    for t, i in zip(totals, infected):
        if i > t:
            return None

    area_info = _parse_field_area(text)
    if area_info is None:
        return None
    area_val, area_unit = area_info
    if area_val <= 0:
        return None

    radius = _parse_quadrat_radius_m(text)
    quads = _assign_quadrants(text, len(totals))

    samples = [
        FieldSample(
            quadrant_id=quads[j] if j < len(quads) else None,
            total_plants=totals[j],
            infected_plants=infected[j],
        )
        for j in range(len(totals))
    ]

    try:
        return FieldSamplingReport(
            total_field_area=area_val,
            area_unit=area_unit,
            quadrat_radius_m=radius,
            samples=samples,
        )
    except Exception:
        return None