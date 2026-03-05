"""
Tests for part1.py utility functions.
"""

import torch
from unittest.mock import patch, MagicMock
import part1


class MockTokenizer:
    """Mock tokenizer for testing format_training_example."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        result = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        if add_generation_prompt:
            result += "\n[assistant]"
        return result


class TestExtractBoxed:
    """Tests for extract_boxed function."""

    def test_extract_boxed_simple(self):
        result = part1.extract_boxed(r"The answer is \boxed{42}")
        assert result == "42"

    def test_extract_boxed_nested(self):
        result = part1.extract_boxed(r"\boxed{\frac{1}{2}}")
        assert result == r"\frac{1}{2}"

    def test_extract_boxed_multiple(self):
        result = part1.extract_boxed(r"First \boxed{3} then \boxed{5}")
        assert result == "5"

    def test_extract_boxed_none(self):
        result = part1.extract_boxed("No boxed answer")
        assert result is None

    def test_extract_boxed_with_spaces(self):
        result = part1.extract_boxed(r"Answer: \boxed{ 123 }")
        assert result == " 123 "

    def test_extract_boxed_deeply_nested(self):
        result = part1.extract_boxed(r"\boxed{outer \boxed{inner} still outer}")
        assert result == "outer \\boxed{inner} still outer"


class TestExtractModelAnswer:
    """Tests for extract_model_answer function."""

    def test_extract_model_answer_boxed(self):
        result = part1.extract_model_answer(r"The answer is \boxed{42}")
        assert result == "42"

    def test_extract_model_answer_answer_colon(self):
        result = part1.extract_model_answer("After calculation, Answer: 42")
        assert result == "42"

    def test_extract_model_answer_final_answer(self):
        result = part1.extract_model_answer("Final Answer: $100")
        assert result == "100"

    def test_extract_model_answer_last_number(self):
        result = part1.extract_model_answer("So the total is 15 cups")
        assert result == "15"

    def test_extract_model_answer_negative(self):
        result = part1.extract_model_answer(r"\boxed{-5}")
        assert result == "-5"

    def test_extract_model_answer_with_commas(self):
        result = part1.extract_model_answer("Answer: 1,234")
        assert result == "1234"

    def test_extract_model_answer_none(self):
        result = part1.extract_model_answer("No numbers here!")
        assert result is None

    def test_extract_model_answer_decimal(self):
        result = part1.extract_model_answer(r"\boxed{3.14159}")
        assert result == "3.14159"


class TestExtractGroundTruth:
    """Tests for extract_ground_truth function."""

    def test_extract_ground_truth_hashmarks(self):
        result = part1.extract_ground_truth("blah #### 5", gt_format="hashmarks")
        assert result == "5"

    def test_extract_ground_truth_comma(self):
        result = part1.extract_ground_truth("#### 1,234", gt_format="hashmarks")
        assert result == "1234"

    def test_extract_ground_truth_no_marker(self):
        result = part1.extract_ground_truth("42", gt_format="hashmarks")
        assert result == "42"

    def test_extract_ground_truth_boxed(self):
        result = part1.extract_ground_truth(r"Answer is \boxed{99}", gt_format="boxed")
        assert result == "99"

    def test_extract_ground_truth_with_spaces(self):
        result = part1.extract_ground_truth("####   42  ", gt_format="hashmarks")
        assert result == "42"


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_normalize_answer_whitespace(self):
        result = part1.normalize_answer("  42  ")
        assert result == "42"

    def test_normalize_answer_commas(self):
        result = part1.normalize_answer("1,234,567")
        assert result == "1234567"

    def test_normalize_answer_dollar(self):
        result = part1.normalize_answer("$100")
        assert result == "100"

    def test_normalize_answer_period(self):
        result = part1.normalize_answer("42.")
        assert result == "42"

    def test_normalize_answer_combined(self):
        result = part1.normalize_answer(" $1,234. ")
        assert result == "1234"


class TestAnswersMatch:
    """Tests for answers_match function."""

    def test_answers_match_exact(self):
        assert part1.answers_match("42", "42") is True

    def test_answers_match_different(self):
        assert part1.answers_match("42", "43") is False

    def test_answers_match_none_predicted(self):
        assert part1.answers_match(None, "42") is False

    def test_answers_match_normalized(self):
        assert part1.answers_match("$1,234", "1234") is True

    def test_answers_match_with_spaces(self):
        assert part1.answers_match(" 42 ", "42") is True


class TestFormatTrainingExample:
    """Tests for format_training_example function."""

    def test_format_training_example(self):
        tokenizer = MockTokenizer()
        question = "What is 2 + 2?"
        answer = "2 + 2 = 4 #### 4"

        result = part1.format_training_example(question, answer, tokenizer)

        assert "[system]" in result
        assert part1.SYSTEM_PROMPT in result
        assert "[user]" in result
        assert question in result
        assert "[assistant]" in result
        assert r"\boxed{4}" in result
        assert "####" not in result

    def test_format_training_example_preserves_reasoning(self):
        tokenizer = MockTokenizer()
        question = "Calculate 3 * 5"
        answer = "3 times 5 equals 15. #### 15"

        result = part1.format_training_example(question, answer, tokenizer)

        assert "3 times 5 equals 15" in result
        assert r"\boxed{15}" in result


class TestLoadGSM8KTestFormat:
    """Tests for load_gsm8k_test function format."""

    @patch("part1.load_dataset")
    def test_load_gsm8k_test_format(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 3
        mock_dataset.__getitem__ = lambda self, idx: {
            "question": f"Question {idx}",
            "answer": f"Answer {idx} #### {idx}",
        }
        mock_load_dataset.return_value = mock_dataset

        result = part1.load_gsm8k_test(num_samples=2)

        assert len(result) == 2
        assert result[0]["question"] == "Question 0"
        assert result[0]["answer"] == "Answer 0 #### 0"
        assert result[1]["question"] == "Question 1"

        mock_load_dataset.assert_called_once_with("gsm8k", "main", split="test")

    @patch("part1.load_dataset")
    def test_load_gsm8k_test_respects_limit(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 100
        mock_dataset.__getitem__ = lambda self, idx: {
            "question": f"Q{idx}",
            "answer": f"A{idx}",
        }
        mock_load_dataset.return_value = mock_dataset

        result = part1.load_gsm8k_test(num_samples=5)

        assert len(result) == 5


class TestConstants:
    """Tests for module constants."""

    def test_model_name(self):
        assert part1.MODEL_NAME == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_seed(self):
        assert part1.SEED == 42

    def test_output_dir(self):
        assert part1.OUTPUT_DIR == "outputs"

    def test_system_prompt_contains_boxed(self):
        assert "boxed" in part1.SYSTEM_PROMPT.lower()

    def test_system_prompt_requests_step_by_step(self):
        prompt_lower = part1.SYSTEM_PROMPT.lower()
        assert "step" in prompt_lower


class TestDeviceDetection:
    """Tests for get_device function."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_cuda(self, mock_cuda):
        result = part1.get_device()
        assert result == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_get_device_mps(self, mock_mps, mock_cuda):
        result = part1.get_device()
        assert result == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_get_device_cpu(self, mock_mps, mock_cuda, capsys):
        result = part1.get_device()
        assert result == "cpu"


class TestSetSeed:
    """Tests for set_seed function."""

    @patch("torch.manual_seed")
    @patch("random.seed")
    @patch("numpy.random.seed")
    def test_set_seed_calls_all(self, mock_np_seed, mock_random_seed, mock_torch_seed):
        part1.set_seed(123)

        mock_torch_seed.assert_called_once_with(123)
        mock_random_seed.assert_called_once_with(123)
        mock_np_seed.assert_called_once_with(123)


class TestBuildFewShotPrompts:
    """Tests for build_few_shot_prompts function."""

    def test_build_few_shot_prompts_contains_examples(self):
        tokenizer = MockTokenizer()
        questions = ["Test question?"]
        result = part1.build_few_shot_prompts(
            tokenizer, questions, part1.FEW_SHOT_EXAMPLES, part1.SYSTEM_PROMPT
        )
        assert len(result) == 1
        prompt = result[0]
        assert "15 trees" in prompt
        assert "3 cars" in prompt
        assert "32 chocolates" in prompt
        assert prompt.endswith("\n[assistant]")

    def test_build_few_shot_prompts_multiple_questions(self):
        tokenizer = MockTokenizer()
        questions = ["Q1?", "Q2?", "Q3?"]
        result = part1.build_few_shot_prompts(
            tokenizer, questions, part1.FEW_SHOT_EXAMPLES, part1.SYSTEM_PROMPT
        )
        assert len(result) == 3


class TestLoraConfig:
    """Tests for get_lora_config function."""

    def test_lora_config_defaults(self):
        config = part1.get_lora_config()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_lora_config_custom(self):
        config = part1.get_lora_config(
            r=16, alpha=32, dropout=0.1, target_modules=["q_proj"]
        )
        assert config.r == 16
        assert config.lora_alpha == 32


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters_structure(self):
        # Create a simple model to test structure
        model = torch.nn.Linear(10, 5)
        result = part1.count_parameters(model)
        assert "total" in result
        assert "trainable" in result
        assert "percentage" in result
        assert result["total"] == 55  # 10*5 + 5 bias
        assert result["trainable"] == 55
        assert result["percentage"] == 100.0


class TestBuildPrompts:
    """Tests for build_prompts function."""

    def test_build_prompts(self):
        result = part1.build_prompts(MockTokenizer(), ["What is 2+2?"])
        assert len(result) == 1
        prompt = result[0]
        assert "[system]" in prompt
        assert "[user]" in prompt
        assert "[assistant]" in prompt
        assert "What is 2+2?" in prompt


class TestMajorityVote:
    """Tests for majority_vote function."""

    def test_majority_vote_clear_winner(self):
        result = part1.majority_vote(["5", "5", "3", "5", "7"])
        assert result == "5"

    def test_majority_vote_all_same(self):
        result = part1.majority_vote(["42", "42", "42"])
        assert result == "42"

    def test_majority_vote_with_nones(self):
        result = part1.majority_vote([None, "5", None, "5", "3"])
        assert result == "5"

    def test_majority_vote_all_none(self):
        result = part1.majority_vote([None, None, None])
        assert result is None

    def test_majority_vote_single(self):
        result = part1.majority_vote(["10"])
        assert result == "10"
