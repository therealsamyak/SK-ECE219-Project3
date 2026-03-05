"""Unit tests for Part 1 answer extraction functions."""

from unittest.mock import Mock

from transformers import AutoTokenizer

from part1 import (
    extract_boxed,
    extract_model_answer,
    extract_ground_truth,
    normalize_answer,
    answers_match,
    format_training_example,
)


class TestExtractBoxed:
    """Tests for extract_boxed function."""

    def test_extract_boxed_simple(self):
        """Test extracting simple boxed answer."""
        text = r"The answer is \boxed{42}"
        result = extract_boxed(text)
        assert result == "42"

    def test_extract_boxed_nested(self):
        """Test extracting boxed answer with nested braces."""
        text = r"\boxed{\frac{1}{2}}"
        result = extract_boxed(text)
        assert result == r"\frac{1}{2}"

    def test_extract_boxed_multiple(self):
        """Test extracting the LAST boxed answer when multiple exist."""
        text = r"First \boxed{3} then \boxed{5}"
        result = extract_boxed(text)
        assert result == "5"

    def test_extract_boxed_none(self):
        """Test when no boxed answer exists."""
        text = r"No boxed answer"
        result = extract_boxed(text)
        assert result is None

    def test_extract_boxed_complex_nested(self):
        """Test extracting boxed with complex nested braces."""
        text = r"\boxed{\frac{\frac{1}{2}}{3}}"
        result = extract_boxed(text)
        assert result == r"\frac{\frac{1}{2}}{3}"

    def test_extract_boxed_whitespace(self):
        """Test extracting boxed with whitespace."""
        text = r"\boxed{  42  }"
        result = extract_boxed(text)
        assert result == "  42  "

    def test_extract_boxed_escaped_backslash(self):
        """Test that escaped backslash doesn't trigger boxed detection."""
        text = r"This is \\\boxed{not detected} but \boxed{42} is"
        result = extract_boxed(text)
        assert result == "42"


class TestExtractModelAnswer:
    """Tests for extract_model_answer function."""

    def test_extract_model_answer_boxed(self):
        """Test extracting answer from boxed format."""
        text = r"Step 1: do this. Step 2: do that. The answer is \boxed{42}."
        result = extract_model_answer(text)
        assert result == "42"

    def test_extract_model_answer_answer_colon(self):
        """Test extracting answer from 'Answer:' pattern."""
        text = "After some calculation, the result is clear. Answer: 42"
        result = extract_model_answer(text)
        assert result == "42"

    def test_extract_model_answer_answer_colon_with_spaces(self):
        """Test extracting answer from 'answer:' with various spacing."""
        text = "so the total is answer:  42"
        result = extract_model_answer(text)
        assert result == "42"

    def test_extract_model_answer_answer_with_dollar(self):
        """Test extracting answer from 'Answer: $42' pattern."""
        text = "The final value is Answer: $42"
        result = extract_model_answer(text)
        assert result == "42"

    def test_extract_model_answer_final_answer(self):
        """Test extracting answer from 'Final Answer:' pattern."""
        text = "After working through the problem, Final Answer: 42"
        result = extract_model_answer(text)
        assert result == "42"

    def test_extract_model_answer_last_number(self):
        """Test extracting last number as fallback."""
        text = "So the total is 15 cups"
        result = extract_model_answer(text)
        assert result == "15"

    def test_extract_model_answer_last_number_with_comma(self):
        """Test extracting last number with comma."""
        text = "The result is 1,234 items"
        result = extract_model_answer(text)
        assert result == "1,234"

    def test_extract_model_answer_negative_number(self):
        """Test extracting negative number."""
        text = "The difference is -5"
        result = extract_model_answer(text)
        assert result == "-5"

    def test_extract_model_answer_decimal(self):
        """Test extracting decimal number."""
        text = "The average is 3.14"
        result = extract_model_answer(text)
        assert result == "3.14"

    def test_extract_model_answer_none(self):
        """Test when no answer can be extracted."""
        text = "This response has no numbers or answer markers."
        result = extract_model_answer(text)
        assert result is None

    def test_extract_model_answer_precedence_boxed(self):
        """Test that boxed has precedence over other patterns."""
        text = r"The answer is \boxed{42} but there is also Answer: 99"
        result = extract_model_answer(text)
        assert result == "42"

    def test_extract_model_answer_precedence_answer_colon(self):
        """Test that Answer: has precedence over last number."""
        text = "Answer: 42 and there is also 99"
        result = extract_model_answer(text)
        assert result == "42"


class TestExtractGroundTruth:
    """Tests for extract_ground_truth function."""

    def test_extract_ground_truth_hashmarks(self):
        """Test extracting ground truth from #### format."""
        text = r"blah #### 5"
        result = extract_ground_truth(text, "hashmarks")
        assert result == "5"

    def test_extract_ground_truth_hashmarks_with_comma(self):
        """Test extracting ground truth from #### format with comma."""
        text = "#### 1,234"
        result = extract_ground_truth(text, "hashmarks")
        assert result == "1234"

    def test_extract_ground_truth_hashmarks_with_spaces(self):
        """Test extracting ground truth from #### format with spaces."""
        text = "The answer is ####   42"
        result = extract_ground_truth(text, "hashmarks")
        assert result == "42"

    def test_extract_ground_truth_boxed(self):
        """Test extracting ground truth from boxed format."""
        text = r"The solution is \boxed{42}"
        result = extract_ground_truth(text, "boxed")
        assert result == "42"

    def test_extract_ground_truth_boxed_nested(self):
        """Test extracting ground truth from boxed format with nested braces."""
        text = r"\boxed{\frac{1}{2}}"
        result = extract_ground_truth(text, "boxed")
        assert result == r"\frac{1}{2}"

    def test_extract_ground_truth_boxed_fallback(self):
        """Test extracting ground truth from boxed format when no boxed found."""
        text = "The answer is 42"
        result = extract_ground_truth(text, "boxed")
        assert result == "The answer is 42"

    def test_extract_ground_truth_unknown_format(self):
        """Test extracting ground truth with unknown format."""
        text = "42"
        result = extract_ground_truth(text, "unknown")
        assert result == "42"


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_normalize_answer_commas(self):
        """Test removing commas."""
        assert normalize_answer("1,234") == "1234"

    def test_normalize_answer_dollar_sign(self):
        """Test removing leading $."""
        assert normalize_answer("$42") == "42"

    def test_normalize_answer_whitespace(self):
        """Test stripping whitespace."""
        assert normalize_answer("  42  ") == "42"

    def test_normalize_answer_trailing_period(self):
        """Test stripping trailing period."""
        assert normalize_answer("42.") == "42"

    def test_normalize_answer_combined(self):
        """Test multiple transformations combined."""
        assert normalize_answer("  $1,234.  ") == "1234"

    def test_normalize_answer_none(self):
        """Test normalizing None."""
        assert normalize_answer(None) == ""

    def test_normalize_answer_empty(self):
        """Test normalizing empty string."""
        assert normalize_answer("") == ""

    def test_normalize_answer_no_changes_needed(self):
        """Test answer that needs no changes."""
        assert normalize_answer("42") == "42"

    def test_normalize_answer_negative(self):
        """Test negative number."""
        assert normalize_answer("-42") == "-42"

    def test_normalize_answer_decimal(self):
        """Test decimal number."""
        assert normalize_answer("3.14") == "3.14"

    def test_normalize_answer_internal_periods(self):
        """Test that internal periods are preserved."""
        assert normalize_answer("3.14") == "3.14"


class TestAnswersMatch:
    """Tests for answers_match function."""

    def test_answers_match_exact(self):
        """Test exact match."""
        assert answers_match("42", "42") is True

    def test_answers_match_with_commas(self):
        """Test match with comma differences."""
        assert answers_match("1,234", "1234") is True

    def test_answers_match_with_dollar(self):
        """Test match with $ sign difference."""
        assert answers_match("$42", "42") is True

    def test_answers_match_with_whitespace(self):
        """Test match with whitespace differences."""
        assert answers_match("  42  ", "42") is True

    def test_answers_match_with_trailing_period(self):
        """Test match with trailing period difference."""
        assert answers_match("42.", "42") is True

    def test_answers_match_combined_differences(self):
        """Test match with multiple differences."""
        assert answers_match("  $1,234.  ", "1234") is True

    def test_answers_match_no_match(self):
        """Test no match."""
        assert answers_match("42", "99") is False

    def test_answers_match_predicted_none(self):
        """Test when predicted is None."""
        assert answers_match(None, "42") is False

    def test_answers_match_normalized_mismatch(self):
        """Test that normalization is applied before comparison."""
        assert answers_match("42.", "99") is False

    def test_answers_match_both_none_like(self):
        """Test when both are None-like."""
        assert answers_match(None, "") is True


class TestFormatTrainingExample:
    """Tests for format_training_example function."""

    def test_format_training_example_contains_boxed(self):
        """Test that formatted example contains \\boxed{}."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        question = "What is 2 + 2?"
        answer = "First, 2 + 2 = 4. So the answer is 4"

        result = format_training_example(question, answer, tokenizer)

        assert r"\boxed{4}" in result

    def test_format_training_example_contains_system_prompt(self):
        """Test that formatted example contains system prompt text."""
        from part1 import SYSTEM_PROMPT

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        question = "What is 2 + 2?"
        answer = "First, 2 + 2 = 4. So the answer is 4"

        result = format_training_example(question, answer, tokenizer)

        assert SYSTEM_PROMPT in result

    def test_format_training_example_with_complex_answer(self):
        """Test formatting with multi-step reasoning."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        question = "What is 3 * 5?"
        answer = "Step 1: 3 * 5 = 15. So the answer is 15"

        result = format_training_example(question, answer, tokenizer)

        assert r"\boxed{15}" in result
        assert "Step 1: 3 * 5 = 15." in result
        assert "So the answer is" not in result

    def test_format_training_example_with_hashmarks(self):
        """Test that #### format is removed and replaced with \\boxed{}."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        question = "What is 10 / 2?"
        answer = "10 divided by 2 equals 5. #### 5"

        result = format_training_example(question, answer, tokenizer)

        assert r"\boxed{5}" in result
        assert "#### 5" not in result


class TestLoadGSM8KTest:
    """Tests for load_gsm8k_test function (mocked)."""

    def test_load_gsm8k_test_format(self, monkeypatch):
        """Test that load_gsm8k_test returns correct dict structure."""
        from part1 import load_gsm8k_test

        mock_dataset = Mock()
        mock_row1 = {"question": "Test question 1?", "answer": "Test answer 1"}
        mock_row2 = {"question": "Test question 2?", "answer": "Test answer 2"}
        mock_selected = Mock()
        mock_selected.__iter__ = Mock(return_value=iter([mock_row1, mock_row2]))
        mock_selected.__len__ = Mock(return_value=2)
        mock_dataset.select.return_value = mock_selected
        mock_dataset.__len__ = Mock(return_value=10)

        def mock_load_dataset(*args, **kwargs):
            return mock_dataset

        import part1

        original_load_dataset = part1.load_dataset
        monkeypatch.setattr(part1, "load_dataset", mock_load_dataset)

        try:
            result = load_gsm8k_test(num_samples=2)

            assert len(result) == 2
            assert result[0]["question"] == "Test question 1?"
            assert result[0]["answer"] == "Test answer 1"
            assert result[1]["question"] == "Test question 2?"
            assert result[1]["answer"] == "Test answer 2"
        finally:
            monkeypatch.setattr(part1, "load_dataset", original_load_dataset)
