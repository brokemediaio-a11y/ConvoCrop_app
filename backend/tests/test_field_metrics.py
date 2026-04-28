"""Unit tests for field sampling metrics (no model required)."""
import unittest

from app.field_metrics import (
    classify_field_severity_band,
    compute_field_metrics,
    metrics_result_to_state_display,
)
from app.inference import _build_field_context_prefix, maybe_append_sampling_guidance
from app.config import OFF_TOPIC_RESPONSE
from app.schemas import FieldSample, FieldSamplingReport, RatingCount, FieldMetricsState


class TestFieldMetrics(unittest.TestCase):
    def test_incidence_and_average(self):
        report = FieldSamplingReport(
            total_field_area=10.0,
            area_unit="ha",
            samples=[
                FieldSample(total_plants=20, infected_plants=8),
                FieldSample(total_plants=20, infected_plants=10),
            ],
        )
        r = compute_field_metrics(report)
        self.assertEqual(r.per_sample_incidence_pct, [40.0, 50.0])
        self.assertEqual(r.average_incidence_pct, 45.0)
        self.assertEqual(r.num_samples, 2)
        self.assertAlmostEqual(r.estimated_infected_area, 4.5, places=3)
        self.assertEqual(r.estimated_infected_area_unit, "ha")
        self.assertIn("45.0%", r.narrative)

    def test_quadrat_area(self):
        report = FieldSamplingReport(
            total_field_area=1.0,
            area_unit="ha",
            quadrat_radius_m=0.5,
            samples=[FieldSample(total_plants=10, infected_plants=1)],
        )
        r = compute_field_metrics(report)
        self.assertIsNotNone(r.quadrat_area_m2)
        self.assertAlmostEqual(r.quadrat_area_m2, 3.14159 * 0.25, places=2)

    def test_severity_ratings(self):
        report = FieldSamplingReport(
            total_field_area=5000.0,
            area_unit="m2",
            samples=[
                FieldSample(
                    total_plants=4,
                    infected_plants=4,
                    rating_counts=[
                        RatingCount(rating=5, count=2),
                        RatingCount(rating=0, count=2),
                    ],
                )
            ],
        )
        r = compute_field_metrics(report)
        self.assertIsNotNone(r.average_severity_pct)
        self.assertGreater(r.average_severity_pct, 0)

    def test_severity_weight_mapping(self):
        report = FieldSamplingReport(
            total_field_area=1.0,
            area_unit="ha",
            samples=[
                FieldSample(
                    total_plants=10,
                    infected_plants=10,
                    rating_counts=[RatingCount(rating=1, count=10)],
                )
            ],
        )
        r = compute_field_metrics(report)
        self.assertEqual(r.average_severity_pct, 20.0)

    def test_classify_field_severity_band_thresholds(self):
        self.assertEqual(classify_field_severity_band(0.0), "healthy")
        self.assertEqual(classify_field_severity_band(18.5), "mild")
        self.assertEqual(classify_field_severity_band(45.0), "moderate")
        self.assertEqual(classify_field_severity_band(78.0), "severe")

    def test_metrics_result_to_state_display(self):
        report = FieldSamplingReport(
            total_field_area=2.0,
            area_unit="acre",
            samples=[FieldSample(total_plants=100, infected_plants=25)],
        )
        r = compute_field_metrics(report)
        s = metrics_result_to_state_display(r)
        self.assertIn("acre", s)

    def test_kanal_unit_narrative(self):
        report = FieldSamplingReport(
            total_field_area=4.0,
            area_unit="kanal",
            samples=[FieldSample(total_plants=50, infected_plants=10)],
        )
        r = compute_field_metrics(report)
        self.assertEqual(r.estimated_infected_area_unit, "kanal")
        self.assertIn("kanal", r.narrative.lower())

    def test_invalid_infected_gt_total(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            FieldSample(total_plants=5, infected_plants=10)


class TestSamplingGuidance(unittest.TestCase):
    def test_appends_on_yield_question(self):
        r, app, st, cta = maybe_append_sampling_guidance(
            "What yield loss should I expect?",
            "Some answer.",
            None,
        )
        self.assertTrue(app)
        self.assertTrue(cta)
        self.assertTrue(st.sampling_guidance_offered)
        self.assertEqual(r, "Some answer.")

    def test_skips_second_time(self):
        prev = FieldMetricsState(sampling_guidance_offered=True)
        r, app, st, cta = maybe_append_sampling_guidance(
            "Tell me about prevention",
            "Advice.",
            prev,
        )
        self.assertFalse(app)
        self.assertFalse(cta)
        self.assertEqual(r, "Advice.")

    def test_remind_overrides_skip(self):
        prev = FieldMetricsState(sampling_guidance_offered=True)
        r, app, st, cta = maybe_append_sampling_guidance(
            "remind me the sampling steps",
            "Advice.",
            prev,
        )
        self.assertTrue(app)
        self.assertTrue(cta)
        self.assertTrue(st.sampling_guidance_offered)

    def test_no_append_on_off_topic(self):
        r, app, st, cta = maybe_append_sampling_guidance(
            "what about yield",
            OFF_TOPIC_RESPONSE,
            None,
        )
        self.assertFalse(app)
        self.assertFalse(cta)
        self.assertEqual(r, OFF_TOPIC_RESPONSE)

    def test_no_cta_when_field_metrics_in_state(self):
        prev = FieldMetricsState(
            sampling_guidance_offered=False,
            last_avg_incidence_pct=12.5,
            last_num_samples=4,
        )
        r, app, st, cta = maybe_append_sampling_guidance(
            "What yield loss should I expect?",
            "Some answer.",
            prev,
        )
        self.assertFalse(app)
        self.assertFalse(cta)
        self.assertEqual(r, "Some answer.")

    def test_cta_on_prevention_measures_even_if_prevent_typo(self):
        """'measures' / 'spread' match even when 'preventive' is mistyped."""
        r, app, st, cta = maybe_append_sampling_guidance(
            "what oreventive measures can i take to stop the spread of this disease?",
            "Avoid shade and rotate fungicides.",
            None,
        )
        self.assertTrue(app)
        self.assertTrue(cta)
        self.assertEqual(r, "Avoid shade and rotate fungicides.")

    def test_field_context_prefix_exact_format(self):
        st = FieldMetricsState(
            last_avg_incidence_pct=45.0,
            last_num_samples=8,
            last_estimated_infected_area_ha=1.2,
            last_field_area_ha=11.5,
            last_avg_severity_pct=36.0,
            field_severity_tier="moderate",
        )
        prefix = _build_field_context_prefix("blast", st)
        self.assertEqual(
            prefix,
            "[Field Context: Disease: rice blast. Field incidence: 45.0%. "
            "Samples: 8. Estimated infected area: 1.2 ha out of 11.5 ha. "
            "Field severity index: 36.0%. Field severity: moderate.]",
        )


if __name__ == "__main__":
    unittest.main()
