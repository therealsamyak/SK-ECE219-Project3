"""Quick unit tests for part1.py fixes - no model loading"""

import sys

sys.path.insert(0, ".")

from part1 import compare_answers_numeric, classify_failure


def test_compare_answers_numeric():
    # Test numeric comparison
    assert compare_answers_numeric("33", "33")
    assert compare_answers_numeric("33.00", "33")  # Key fix
    assert not compare_answers_numeric("33.001", "33")
    assert compare_answers_numeric("1,000", "1000")
    assert not compare_answers_numeric(None, "33")
    assert not compare_answers_numeric("abc", "33")  # Falls back to string
    print("✓ compare_answers_numeric tests passed")


def test_classify_failure():
    # Test reasoning vs arithmetic classification
    # Case 1: No arithmetic errors in response, wrong answer = reasoning_error
    result = classify_failure(
        question="What is the ratio?",
        response="The answer is 32.5",  # No arithmetic expressions
        extracted_answer="32.5",
        ground_truth="109",
    )
    assert result == "reasoning_error", f"Expected reasoning_error, got {result}"

    # Case 2: Arithmetic error in response = arithmetic_error
    result = classify_failure(
        question="Calculate sum",
        response="15 + 25 = 7",  # Wrong arithmetic
        extracted_answer="7",
        ground_truth="40",
    )
    assert result == "arithmetic_error", f"Expected arithmetic_error, got {result}"
    print("✓ classify_failure tests passed")


if __name__ == "__main__":
    test_compare_answers_numeric()
    test_classify_failure()
    print("\n✓ All tests passed!")
