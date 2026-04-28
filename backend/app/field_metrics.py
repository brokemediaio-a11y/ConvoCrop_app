"""Deterministic field sampling metrics (incidence, severity index, infected area)."""
from __future__ import annotations

import math
from typing import List, Optional

from app.schemas import AreaUnit, FieldMetricsResult, FieldSamplingReport

_M2_PER_ACRE = 4046.8564224
_M2_PER_HA = 10000.0

MAX_RATING = 5
_RATING_TO_SEVERITY_WEIGHT = {
    0: 0.0,   # healthy
    1: 20.0,  # mild
    2: 20.0,  # mild
    3: 50.0,  # moderate
    4: 85.0,  # severe
    5: 85.0,  # severe
}



def _format_area(value: float, unit: AreaUnit) -> str:
    if unit == "m2":
        return f"{value:.1f} square meters"
    if unit == "kanal":
        return f"{value:.2f} kanal"
    if unit == "marla":
        return f"{value:.2f} marla"
    if unit == "acre":
        return f"{value:.2f} acres"
    if unit == "ha":
        return f"{value:.2f} hectares"
    return f"{value:.2f}"


def compute_field_metrics(report: FieldSamplingReport) -> FieldMetricsResult:
    """
    Compute per-sample incidence, average incidence, optional severity index,
    estimated infected field area, and optional quadrat area.
    """
    samples = report.samples
    per_inc: List[float] = []
    per_sev: Optional[List[float]] = None
    rated_only: List[float] = []

    for s in samples:
        t = s.total_plants
        i = s.infected_plants
        inc = (i / t) * 100.0 if t > 0 else 0.0
        per_inc.append(round(inc, 2))

    n = len(per_inc)
    avg_inc = sum(per_inc) / n if n else 0.0

    avg_sev: Optional[float] = None
    if any(s.rating_counts for s in samples):
        per_sev = []
        for s in samples:
            if s.rating_counts:
                t = s.total_plants
                weighted_sum = sum(
                    _RATING_TO_SEVERITY_WEIGHT.get(rc.rating, 0.0) * rc.count
                    for rc in s.rating_counts
                )
                sv = round((weighted_sum / t), 2) if t > 0 else 0.0
                per_sev.append(sv)
                rated_only.append(sv)
            else:
                per_sev.append(0.0)
        avg_sev = (
            round(sum(rated_only) / len(rated_only), 2) if rated_only else None
        )

    total_area = report.total_field_area
    unit = report.area_unit
    est_infected = round((avg_inc / 100.0) * total_area, 4)

    q_area: Optional[float] = None
    if report.quadrat_radius_m is not None:
        r = report.quadrat_radius_m
        q_area = round(math.pi * r * r, 4)

    narrative = _build_narrative(
        per_inc=per_inc,
        avg_inc=avg_inc,
        est_infected=est_infected,
        unit=unit,
        total_area=total_area,
        n=n,
        q_area_m2=q_area,
        per_sev=per_sev,
        avg_sev=avg_sev,
    )

    return FieldMetricsResult(
        per_sample_incidence_pct=per_inc,
        average_incidence_pct=round(avg_inc, 2),
        num_samples=n,
        estimated_infected_area=est_infected,
        estimated_infected_area_unit=unit,
        total_field_area=total_area,
        quadrat_area_m2=q_area,
        per_sample_severity_pct=per_sev,
        average_severity_pct=avg_sev,
        narrative=narrative,
    )


def _incidence_label(pct: float) -> str:
    """Human-friendly label for an incidence percentage."""
    if pct == 0:
        return "no infection detected"
    if pct < 10:
        return "low"
    if pct < 30:
        return "moderate"
    if pct < 60:
        return "high"
    return "very high"


def _build_narrative(
    *,
    per_inc: List[float],
    avg_inc: float,
    est_infected: float,
    unit: AreaUnit,
    total_area: float,
    n: int,
    q_area_m2: Optional[float],
    per_sev: Optional[List[float]],
    avg_sev: Optional[float],
) -> str:
    lines: List[str] = []

    # Opening
    lines.append(
        f"Based on your {n} sample{'s' if n != 1 else ''}, "
        "here is what the numbers tell us about your field."
    )
    lines.append("")

    # Per-sample breakdown
    lines.append("**Infection rate per sample:**")
    for idx, p in enumerate(per_inc, start=1):
        lines.append(
            f"  Sample {idx}: {p:.1f}% of plants showed infection "
            f"({_incidence_label(p)})."
        )
    lines.append("")

    # Average
    label = _incidence_label(avg_inc)
    lines.append(
        f"Overall, about {avg_inc:.1f}% of plants are infected on average "
        f"across your samples — that is considered **{label}** incidence. "
        "Keep in mind this depends on how spread out your sample spots were."
    )
    lines.append("")

    # Estimated infected area
    lines.append(
        f"Your field is about {_format_area(total_area, unit)}. "
        f"If infection is distributed similarly to your samples, "
        f"roughly {_format_area(est_infected, unit)} may be affected. "
        "This is a planning estimate, not a precise boundary."
    )

    # Quadrat area
    if q_area_m2 is not None:
        lines.append("")
        lines.append(
            f"Each circular quadrat covers about {q_area_m2:.2f} square meters "
            "based on the radius you provided."
        )

    # Severity index
    if per_sev is not None and avg_sev is not None:
        lines.append("")
        if avg_sev < 20:
            sev_word = "mild"
        elif avg_sev < 50:
            sev_word = "moderate"
        else:
            sev_word = "severe"
        lines.append(
            f"Using the 0-to-5 severity scores you recorded, the average severity "
            f"index comes out to about {avg_sev:.1f}%, which suggests **{sev_word}** "
            "symptoms on average across the sampled plots."
        )

    # Closing
    lines.append("")
    lines.append(
        "Feel free to ask about treatment options or expected yield impact — "
        "these field estimates will stay available throughout this conversation."
    )
    return "\n".join(lines)


def metrics_result_to_state_display(result: FieldMetricsResult) -> str:
    """Compact human-readable infected area for FieldMetricsState."""
    return _format_area(result.estimated_infected_area, result.estimated_infected_area_unit)


def convert_area_to_ha(value: float, unit: AreaUnit) -> float:
    """Convert an area value from any supported unit to hectares."""
    if unit == "ha":
        return float(value)
    if unit == "acre":
        return float(value) * (_M2_PER_ACRE / _M2_PER_HA)
    if unit == "m2":
        return float(value) / _M2_PER_HA
    # Approximate local units to square meters:
    # 1 kanal = 505.857052 m2, 1 marla = 25.29285264 m2
    if unit == "kanal":
        return float(value) * (505.857052 / _M2_PER_HA)
    if unit == "marla":
        return float(value) * (25.29285264 / _M2_PER_HA)
    return float(value)


def classify_field_severity_band(
    incidence_pct: float = 0.0,
    severity_idx: float = 0.0,
    disease: str = "blast",
    **_kwargs,
) -> str:
    """
    Disease-aware field severity classification using IRRI-aligned thresholds.
    Tiers: low / moderate / high / critical (matching VQA training data).
    """
    _THRESHOLDS = {
        "blast": {
            "low":      {"max_inc": 15, "max_sev": 20},
            "moderate": {"max_inc": 35, "max_sev": 45},
            "high":     {"max_inc": 60, "max_sev": 70},
        },
        "blight": {
            "low":      {"max_inc": 10, "max_sev": 15},
            "moderate": {"max_inc": 30, "max_sev": 40},
            "high":     {"max_inc": 55, "max_sev": 65},
        },
        "brownspot": {
            "low":      {"max_inc": 20, "max_sev": 20},
            "moderate": {"max_inc": 45, "max_sev": 45},
            "high":     {"max_inc": 70, "max_sev": 65},
        },
    }

    inc = max(0.0, float(incidence_pct))
    sev = max(0.0, float(severity_idx))
    thresholds = _THRESHOLDS.get(disease, _THRESHOLDS["blast"])

    for tier in ("low", "moderate", "high"):
        t = thresholds[tier]
        if inc <= t["max_inc"] and sev <= t["max_sev"]:
            return tier
    return "critical"