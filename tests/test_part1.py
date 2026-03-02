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
