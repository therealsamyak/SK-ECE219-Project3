import json
import pytest

from part2 import (
    test_planner_output,
    PlannerOutput,
    ObservationOutput,
    Executor,
    ReActAgent,
)
import pandas as pd


def test_q18_structured_output():
    """Test that structured output returns valid PlannerOutput."""
    prompt = "The user asks: What is the average age?"

    result = test_planner_output(prompt)

    assert isinstance(result, dict)
    assert "thought" in result
    assert "is_done" in result
    assert "response" in result

    assert isinstance(result["thought"], str)
    assert 10 <= len(result["thought"]) <= 500
    assert isinstance(result["is_done"], bool)
    assert isinstance(result["response"], str)
    assert len(result["response"]) >= 1


def test_planner_output_schema():
    """Test that PlannerOutput schema is correctly defined."""
    assert hasattr(PlannerOutput, "model_fields")
    fields = PlannerOutput.model_fields
    assert "thought" in fields
    assert "is_done" in fields
    assert "response" in fields


def test_observation_output_schema():
    """Test that ObservationOutput schema is correctly defined."""
    assert hasattr(ObservationOutput, "model_fields")
    fields = ObservationOutput.model_fields
    assert "summary" in fields
    assert "extracted_values" in fields
    assert "error_type" in fields


def test_q18_demo_json_exists():
    """Test that q18_demo.json exists and has correct structure."""
    try:
        with open("outputs/q18_demo.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        pytest.skip("q18_demo.json not found - run q18 first")

    assert "model" in data
    assert "quantization" in data
    assert "num_prompts" in data
    assert "prompts" in data
    assert isinstance(data["prompts"], list)
    assert len(data["prompts"]) == 5

    for entry in data["prompts"]:
        assert "prompt" in entry
        assert "output" in entry
        output = entry["output"]
        assert "thought" in output
        assert "is_done" in output
        assert "response" in output


def test_q18_has_is_done_true():
    """Test that at least one prompt in q18_demo.json has is_done=true."""
    try:
        with open("outputs/q18_demo.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        pytest.skip("q18_demo.json not found - run q18 first")

    has_done = any(p["output"]["is_done"] for p in data["prompts"])
    assert has_done, "At least one prompt should have is_done=true"


def test_executor_successful_execution():
    """Test Executor with successful code execution."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    executor = Executor(df)

    stdout, stderr = executor.run("print('hello')\nresult = df['a'].sum()")
    assert "hello" in stdout
    assert stderr == ""
    assert "result" in executor._namespace


def test_executor_timeout_handling():
    """Test Executor timeout with infinite loop."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df, timeout=1)

    stdout, stderr = executor.run("while True: pass")
    assert "TimeoutError" in stderr or "timed out" in stderr.lower()


def test_executor_syntax_error_catching():
    """Test Executor catches syntax errors."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df)

    stdout, stderr = executor.run("invalid syntax here")
    assert "SyntaxError" in stderr


def test_executor_runtime_error_catching():
    """Test Executor catches runtime errors."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df)

    stdout, stderr = executor.run("x = 1 / 0")
    assert "ZeroDivisionError" in stderr or "division by zero" in stderr.lower()


def test_executor_namespace_reset():
    """Test Executor resets namespace between calls."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df)

    executor.run("x = 42")
    assert "x" in executor._namespace

    executor.reset()
    assert "x" not in executor._namespace

    stdout, stderr = executor.run("print(x)")
    assert "NameError" in stderr or "name 'x' is not defined" in stderr


def test_executor_isolated_namespace():
    """Test Executor provides only safe modules in namespace."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df)

    # Check that safe modules are available
    assert "pd" in executor._namespace
    assert "np" in executor._namespace
    assert "df" in executor._namespace
    assert "print" in executor._namespace

    # Check that unsafe modules are not available
    assert "os" not in executor._namespace
    assert "sys" not in executor._namespace
    assert "subprocess" not in executor._namespace


def test_executor_namespace_persistence_within_task():
    """Test that variables persist within a single task (no reset between calls)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df)

    executor.run("x = 10")
    executor.run("y = x * 2")
    executor.run("result = y + 5")

    assert "x" in executor._namespace
    assert "y" in executor._namespace
    assert executor._namespace["result"] == 25


def test_executor_dataframe_access():
    """Test Executor can access DataFrame via df variable."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    executor = Executor(df)

    stdout, stderr = executor.run("print(df.shape)")
    assert "(3, 2)" in stdout

    stdout, stderr = executor.run("print(df['a'].sum())")
    assert "6" in stdout


def test_executor_custom_timeout():
    """Test Executor with custom timeout duration."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    executor = Executor(df, timeout=1)

    stdout, stderr = executor.run("x = 0\nwhile True:\n    x += 1")
    assert "TimeoutError" in stderr or "timed out" in stderr.lower()


def test_coder_observer():
    """Test Coder and Observer methods of ReActAgent."""
    agent = ReActAgent()

    code = agent.coder(
        instruction="Calculate the mean of column A",
        question="What is the mean of column A?",
        context="DataFrame df has column A with numeric values",
    )

    assert isinstance(code, str)
    assert len(code) > 0

    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Coder generated invalid Python code: {e}\nCode:\n{code}")

    code_with_markdown = agent._extract_code_from_markdown(
        "Here's the code:\n```python\nprint('hello')\n```\nDone"
    )
    assert code_with_markdown == "print('hello')"

    code_without_markdown = agent._extract_code_from_markdown("print('hello')")
    assert code_without_markdown == "print('hello')"

    code_with_python_tag = agent._extract_code_from_markdown("```python\nx = 42\n```")
    assert code_with_python_tag == "x = 42"

    obs = agent.observer(stdout="Mean: 42.5", stderr="", instruction="Calculate mean")

    assert isinstance(obs, ObservationOutput)
    assert hasattr(obs, "summary")
    assert hasattr(obs, "extracted_values")
    assert hasattr(obs, "error_type")
    assert isinstance(obs.summary, str)
    assert isinstance(obs.extracted_values, dict)
    assert obs.error_type is None or isinstance(obs.error_type, str)

    obs_empty = agent.observer(stdout="", stderr="", instruction="Calculate something")

    assert isinstance(obs_empty, ObservationOutput)
    assert hasattr(obs_empty, "summary")
    assert hasattr(obs_empty, "extracted_values")
    assert hasattr(obs_empty, "error_type")

    obs_error = agent.observer(
        stdout="",
        stderr="ZeroDivisionError: division by zero",
        instruction="Divide by zero",
    )

    assert isinstance(obs_error, ObservationOutput)
    assert obs_error.error_type is not None or "error" in obs_error.summary.lower()


def test_q20_agent():
    try:
        with open("outputs/q20_accuracy.json") as f:
            accuracy_data = json.load(f)
    except FileNotFoundError:
        pytest.skip("q20_accuracy.json not found - run q20 first")

    assert "accuracy" in accuracy_data
    assert 0.0 <= accuracy_data["accuracy"] <= 1.0
    assert "correct" in accuracy_data
    assert "total" in accuracy_data
    assert accuracy_data["total"] == 10
    assert "results" in accuracy_data
    assert isinstance(accuracy_data["results"], list)
    assert len(accuracy_data["results"]) == 10

    for result in accuracy_data["results"]:
        assert "task_id" in result
        assert "is_correct" in result
        assert "final_answer" in result
        assert "ground_truth" in result

    try:
        with open("outputs/q20_traces.json") as f:
            traces = json.load(f)
    except FileNotFoundError:
        pytest.skip("q20_traces.json not found - run q20 first")

    assert isinstance(traces, list)
    assert len(traces) >= 3

    for trace in traces:
        assert "task_id" in trace
        assert "history" in trace
        assert "final_answer" in trace
        assert isinstance(trace["history"], list)

        for history_entry in trace["history"]:
            assert "thought" in history_entry
            assert "instruction" in history_entry
            assert "code" in history_entry
            assert "observation" in history_entry


def test_react_agent_evaluate_answer():
    agent = ReActAgent()

    predicted = "@mean_fare[34.65]"
    ground_truth = [["mean_fare", "34.65"]]
    assert agent.evaluate_answer(predicted, ground_truth) is True

    predicted = "@mean_fare[34.66]"
    ground_truth = [["mean_fare", "34.65"]]
    assert agent.evaluate_answer(predicted, ground_truth) is True

    predicted = "@mean_fare[35.00]"
    ground_truth = [["mean_fare", "34.65"]]
    assert agent.evaluate_answer(predicted, ground_truth) is False

    predicted = "@mean_fare[34.65], @median_fare[15.00]"
    ground_truth = [["mean_fare", "34.65"], ["median_fare", "15.00"]]
    assert agent.evaluate_answer(predicted, ground_truth) is True

    predicted = "@mean_fare[34.65]"
    ground_truth = [["mean_fare", "34.65"], ["median_fare", "15.00"]]
    assert agent.evaluate_answer(predicted, ground_truth) is False

    predicted = "@name[hello]"
    ground_truth = [["name", "hello"]]
    assert agent.evaluate_answer(predicted, ground_truth) is True

    predicted = "@name[world]"
    ground_truth = [["name", "hello"]]
    assert agent.evaluate_answer(predicted, ground_truth) is False
