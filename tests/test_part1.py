# tests/test_part1.py
import sys

sys.path.insert(0, ".")  # Allow importing part1


class TestDeviceDetection:
    def test_device_detection_returns_valid_device(self):
        """Test that device detection returns cuda/mps/cpu"""
        from part1 import get_device

        device = get_device()
        assert device in ["cuda", "mps", "cpu"]


class TestAnswerExtraction:
    def test_extract_ground_truth_basic(self):
        """Test extracting answer from #### format"""
        from part1 import extract_ground_truth

        assert extract_ground_truth("Some text #### 42") == "42"
        assert extract_ground_truth("Answer is #### 100") == "100"

    def test_extract_ground_truth_negative(self):
        """Test negative numbers"""
        from part1 import extract_ground_truth

        assert extract_ground_truth("Result #### -5") == "-5"

    def test_extract_ground_truth_no_marker(self):
        """Test when no marker present"""
        from part1 import extract_ground_truth

        result = extract_ground_truth("No marker here")
        assert result is None or result == ""


class TestModelLoading:
    def test_load_tokenizer(self):
        """Test tokenizer loads correctly"""
        from part1 import load_tokenizer

        tokenizer = load_tokenizer()
        assert tokenizer is not None
        assert tokenizer.pad_token is not None


class TestGSM8KLoading:
    def test_load_gsm8k_subset(self):
        """Test loading small subset of GSM8K"""
        from part1 import load_gsm8k_test

        test_data = load_gsm8k_test(n_samples=3, seed=42)
        assert len(test_data) == 3
        assert "question" in test_data[0]
        assert "answer" in test_data[0]


class TestFewShotPrompt:
    def test_build_few_shot_prompt_structure(self):
        """Test few-shot prompt has correct structure"""
        from part1 import build_few_shot_prompt

        examples = [{"question": "1+1=?", "answer": "2"}]
        prompt = build_few_shot_prompt("What is 2+2?", examples, n_shots=1)
        assert "1+1" in prompt
        assert "2+2" in prompt


class TestSelfConsistency:
    def test_majority_vote(self):
        """Test majority voting logic"""
        from part1 import majority_vote

        assert majority_vote(["1", "2", "2", "2", "3"]) == "2"
        assert majority_vote(["5", "5", "3"]) == "5"


class TestOutputFiles:
    def test_output_directory_exists(self):
        """Test outputs/ directory is created"""
        import os
        from part1 import ensure_output_dir

        ensure_output_dir()
        assert os.path.exists("outputs")


class TestModelAnswerExtraction:
    def test_extract_boxed_simple(self):
        """Test extracting from \boxed{}"""
        from part1 import extract_boxed

        assert extract_boxed("The answer is \\boxed{42}") == "42"
        assert extract_boxed("\\boxed{100} is correct") == "100"

    def test_extract_boxed_nested(self):
        """Test nested braces"""
        from part1 import extract_boxed

        assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_extract_model_answer_boxed(self):
        """Test model answer extraction prefers boxed"""
        from part1 import extract_model_answer

        assert extract_model_answer("Result: \\boxed{5}") == "5"

    def test_extract_model_answer_fallback(self):
        """Test fallback to last number"""
        from part1 import extract_model_answer

        assert extract_model_answer("First 10, then 20, finally 30") == "30"

    def test_extract_model_answer_with_commas(self):
        """Test number with commas"""
        from part1 import extract_model_answer

        result = extract_model_answer("Total: \\boxed{1,000}")
        assert result == "1000"


class TestParameterCounting:
    def test_count_parameters_structure(self):
        """Test count_parameters returns correct structure"""
        from part1 import count_parameters, load_base_model

        model, _ = load_base_model()
        counts = count_parameters(model)

        assert "total_params" in counts
        assert "trainable_params" in counts
        assert "trainable_pct" in counts
        assert counts["total_params"] > 0

    def test_lora_config_values(self):
        """Test get_lora_config returns correct defaults"""
        from part1 import get_lora_config

        config = get_lora_config()

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestGroundTruthExtraction:
    def test_extract_with_commas(self):
        """Test numbers with commas"""
        from part1 import extract_ground_truth

        assert extract_ground_truth("Total: #### 1,234") == "1234"

    def test_extract_decimal(self):
        """Test decimal numbers"""
        from part1 import extract_ground_truth

        result = extract_ground_truth("Answer: #### 3.5")
        assert result == "3.5"
