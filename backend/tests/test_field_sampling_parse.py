"""Tests for free-text field sampling parser."""
import unittest

from app.field_sampling_parse import (
    looks_like_field_sample_submission,
    try_parse_field_sampling_from_text,
)
from app.field_metrics import compute_field_metrics


USER_STYLE_SAMPLE = """
ok i divided the field in4 parts as you said and took 3 samples form each part
Sample 1 Part 1:
Total plants: 4
Infected plants: 1

Sample 2 part 1:
Total plants: 6
Infected plants: 2

Sample 3 part 1:
Total plants: 5
Infected plants: 3

Sample 1 part 2:
Total plants: 6
Infected plants: 3

Sample 2 part 2:
Total plants: 4
Infected plants: 1

Sample 3 part 2:
Total plants: 6
Infected plants: 2

Sample 1 part 3:
Total plants: 7
Infected plants: 5

Sample 2 part 3:
Total plants: 6
Infected plants: 2

Sample 3 part 3:
Total plants: 4
Infected plants: 2

Sample 1 part 4:
Total plants: 4
Infected plants: 1

Sample 2 part 4:
Total plants: 5
Infected plants: 3

Sample 3 part 4:
Total plants: 6
Infected plants: 5


Total Area: 500 meter square
Area of quadrant: 1 meter radius
"""


class TestFieldSamplingParse(unittest.TestCase):
    def test_looks_like_submission(self):
        self.assertTrue(looks_like_field_sample_submission(USER_STYLE_SAMPLE))

    def test_parse_user_style_message(self):
        report = try_parse_field_sampling_from_text(USER_STYLE_SAMPLE)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(len(report.samples), 12)
        self.assertEqual(report.area_unit, "m2")
        self.assertAlmostEqual(report.total_field_area, 500.0)
        self.assertAlmostEqual(report.quadrat_radius_m or 0, 1.0)
        result = compute_field_metrics(report)
        self.assertEqual(result.num_samples, 12)
        self.assertGreater(result.average_incidence_pct, 0)
        self.assertIn("incidence", result.narrative.lower())

    def test_short_question_not_sampling(self):
        self.assertFalse(looks_like_field_sample_submission("What is rice blast?"))


if __name__ == "__main__":
    unittest.main()
