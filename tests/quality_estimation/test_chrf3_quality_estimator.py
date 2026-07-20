from pytest import approx

from machine.corpora import MultiKeyRef, ScriptureRef
from machine.quality_estimation import ChrF3QualityEstimator, UsabilityLabel
from machine.scripture import VerseRef


def test_chrf3_quality_estimator_txt_files() -> None:
    quality_estimation = ChrF3QualityEstimator(slope=109.6145, intercept=-14.0633)

    confidences = [
        (MultiKeyRef("MAT.txt", [1]), 0.6220749899712906),
        (MultiKeyRef("MAT.txt", [2]), 0.5416165991875662),
        (MultiKeyRef("MRK.txt", [1]), 0.40324150219671917),
    ]

    usability_text_segments, usability_texts = quality_estimation.estimate_quality(confidences)

    # Segment assertions
    assert len(usability_text_segments) == 3

    assert usability_text_segments[0].label == UsabilityLabel.GREEN
    assert usability_text_segments[0].projected_chrf3 == approx(54.12, abs=0.01)
    assert usability_text_segments[0].usability == approx(0.786, abs=0.001)
    assert usability_text_segments[0].confidence == approx(confidences[0][1], abs=0.001)

    assert usability_text_segments[1].label == UsabilityLabel.YELLOW
    assert usability_text_segments[1].projected_chrf3 == approx(45.31, abs=0.01)
    assert usability_text_segments[1].usability == approx(0.691, abs=0.001)
    assert usability_text_segments[1].confidence == approx(confidences[1][1], abs=0.001)

    assert usability_text_segments[2].label == UsabilityLabel.RED
    assert usability_text_segments[2].projected_chrf3 == approx(30.14, abs=0.01)
    assert usability_text_segments[2].usability == approx(0.465, abs=0.001)
    assert usability_text_segments[2].confidence == approx(confidences[2][1], abs=0.001)

    # Text level assertions
    assert len(usability_texts) == 2

    assert usability_texts[0].label == UsabilityLabel.YELLOW
    assert usability_texts[0].projected_chrf3 == approx(49.56, abs=0.01)
    assert usability_texts[0].usability == approx(0.738, abs=0.001)
    assert usability_texts[0].confidence == approx(0.580, abs=0.001)

    assert usability_texts[1].label == UsabilityLabel.RED
    assert usability_texts[1].projected_chrf3 == approx(30.14, abs=0.01)
    assert usability_texts[1].usability == approx(0.465, abs=0.001)
    assert usability_texts[1].confidence == approx(confidences[2][1], abs=0.001)


def test_chrf3_quality_estimator_verses() -> None:
    quality_estimation = ChrF3QualityEstimator(slope=109.6145, intercept=-14.0633)

    confidences = [
        (ScriptureRef(VerseRef(1, 1, 1)), 0.6220749899712906),
        (ScriptureRef(VerseRef(1, 1, 2)), 0.5416165991875662),
        (ScriptureRef(VerseRef(1, 2, 1)), 0.40324150219671917),
    ]

    usability_segments, usability_chapters, usability_books = quality_estimation.estimate_quality(confidences)

    # Segment assertions
    assert len(usability_segments) == 3
    assert usability_segments[0].label == UsabilityLabel.GREEN
    assert usability_segments[0].projected_chrf3 == approx(54.12, abs=0.01)
    assert usability_segments[0].usability == approx(0.786, abs=0.001)
    assert usability_segments[0].confidence == approx(confidences[0][1], abs=0.001)

    assert usability_segments[1].label == UsabilityLabel.YELLOW
    assert usability_segments[1].projected_chrf3 == approx(45.31, abs=0.01)
    assert usability_segments[1].usability == approx(0.691, abs=0.001)
    assert usability_segments[1].confidence == approx(confidences[1][1], abs=0.001)

    assert usability_segments[2].label == UsabilityLabel.RED
    assert usability_segments[2].projected_chrf3 == approx(30.14, abs=0.01)
    assert usability_segments[2].usability == approx(0.465, abs=0.001)
    assert usability_segments[2].confidence == approx(confidences[2][1], abs=0.001)

    # Chapter assertions
    assert len(usability_chapters) == 2
    assert usability_chapters[0].label == UsabilityLabel.YELLOW
    assert usability_chapters[0].projected_chrf3 == approx(49.56, abs=0.01)
    assert usability_chapters[0].usability == approx(0.738, abs=0.001)
    assert usability_chapters[0].confidence == approx(0.580, abs=0.001)

    assert usability_chapters[1].label == UsabilityLabel.RED
    assert usability_chapters[1].projected_chrf3 == approx(30.14, abs=0.01)
    assert usability_chapters[1].usability == approx(0.465, abs=0.001)
    assert usability_chapters[1].confidence == approx(confidences[2][1], abs=0.001)

    # Book assertions
    assert len(usability_books) == 1
    assert usability_books[0].label == UsabilityLabel.RED
    assert usability_books[0].projected_chrf3 == approx(42.28, abs=0.01)
    assert usability_books[0].usability == approx(0.647, abs=0.001)
    assert usability_books[0].confidence == approx(0.514, abs=0.001)
