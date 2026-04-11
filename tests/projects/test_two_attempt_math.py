from __future__ import annotations

from projects.two_attempt_math.agent import CompactStateSelfCorrectionAgent
from projects.two_attempt_math.compute_metrics import summarize_serialized_episodes
from projects.two_attempt_math.env import TwoAttemptMathEnv, strip_hidden_reasoning, two_attempt_math_reward_fn


TASK = {
    "question": "What is 2 + 2?",
    "ground_truth": "4",
    "data_source": "hendrycks_math",
}


def test_attempt_1_correct_terminates_immediately():
    env = TwoAttemptMathEnv(task=TASK)
    initial_obs, initial_info = env.reset()

    assert "Problem:" in initial_obs["state_prompt"]
    assert initial_obs["attempt_index"] == 1
    assert initial_info["attempt_index"] == 1

    next_obs, reward, done, info = env.step("<think>2 + 2 = 4</think>\nThe answer is \\boxed{4}.")

    assert next_obs == {}
    assert reward == 1.0
    assert done is True
    assert info["attempt_1_correct"] is True
    assert info["used_second_attempt"] is False
    assert info["final_correct"] is True


def test_attempt_2_observation_is_compact_state_prompt():
    env = TwoAttemptMathEnv(task=TASK)
    env.reset()

    next_obs, reward, done, info = env.step("<think>2 + 2 = 5</think>\nThe answer is \\boxed{5}.")

    prompt = next_obs["state_prompt"]
    assert reward == 0.0
    assert done is False
    assert next_obs["attempt_index"] == 2
    assert "Original problem:" in prompt
    assert "Your previous attempt:" in prompt
    assert "\\boxed{5}" in prompt
    assert "<think>" not in prompt
    assert "Feedback from the environment:\nincorrect" in prompt
    assert "hint" not in prompt.lower()
    assert "error type" not in prompt.lower()
    assert info["attempt_1_correct"] is False
    assert info["used_second_attempt"] is True
    assert info["final_correct"] is False


def test_second_attempt_reward_is_step_level_value():
    env = TwoAttemptMathEnv(task=TASK)
    env.reset()
    env.step("<think>2 + 2 = 5</think>\nThe answer is \\boxed{5}.")

    next_obs, reward, done, info = env.step("<think>Let me fix it</think>\nThe answer is \\boxed{4}.")

    assert next_obs == {}
    assert reward == 0.6
    assert done is True
    assert info["corrected_on_second_attempt"] is True
    assert info["final_correct"] is True


def test_reward_wrapper_accepts_missing_think_tags():
    output = two_attempt_math_reward_fn(TASK, "The answer is \\boxed{4}.", attempt_index=1)
    assert output.is_correct is True
    assert output.reward == 1.0


def test_strip_hidden_reasoning_keeps_visible_answer_only():
    text = "<think>private reasoning</think>\nVisible answer \\boxed{5}."
    assert strip_hidden_reasoning(text) == "Visible answer \\boxed{5}."


def test_agent_replaces_state_instead_of_accumulating_transcript():
    agent = CompactStateSelfCorrectionAgent()
    agent.trajectory.task = TASK

    first_observation = {
        "state_prompt": "Problem:\nWhat is 2 + 2?",
        "attempt_index": 1,
    }
    second_observation = {
        "state_prompt": "Retry with incorrect feedback and previous attempt.",
        "attempt_index": 2,
    }

    agent.update_from_env(first_observation, 0.0, False, {"attempt_index": 1})
    first_action = agent.update_from_model("<think>wrong</think>\n\\boxed{5}")
    agent.update_from_env(second_observation, 0.0, False, {"attempt_index": 2})
    second_action = agent.update_from_model("Final \\boxed{4}")

    assert first_action.action == "<think>wrong</think>\n\\boxed{5}"
    assert second_action.action == "Final \\boxed{4}"
    assert len(agent.trajectory.steps) == 2
    assert agent.trajectory.is_cumulative() is False
    assert agent.trajectory.steps[0].chat_completions == [{"role": "user", "content": "Problem:\nWhat is 2 + 2?"}]
    assert agent.trajectory.steps[1].chat_completions == [{"role": "user", "content": "Retry with incorrect feedback and previous attempt."}]


def test_offline_metrics_include_primary_names_and_aliases():
    summary = summarize_serialized_episodes(
        [
            {
                "attempt_1_correct": False,
                "final_correct": True,
                "corrected_on_second_attempt": True,
                "attempt_2_response": "\\boxed{4}",
            },
            {
                "attempt_1_correct": True,
                "final_correct": True,
                "corrected_on_second_attempt": False,
                "attempt_2_response": None,
            },
        ]
    )

    assert summary["first_pass_accuracy"] == 0.5
    assert summary["final_accuracy"] == 1.0
    assert summary["correction_rate"] == 1.0
    assert summary["second_attempt_rate"] == 0.5
    assert summary["pass_at_1"] == summary["first_pass_accuracy"]
    assert summary["pass_at_2"] == summary["final_accuracy"]
