from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

from rllm.data.dataset import DatasetRegistry

TRAIN_DATASET_NAME = "two_attempt_math_train"
TEST_DATASET_NAME = "two_attempt_math_test"


def _preprocess_fn(example):
    return {
        "question": example.get("problem", ""),
        "ground_truth": example.get("solution", ""),
        "data_source": "hendrycks_math",
    }


def prepare_two_attempt_math_data():
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    datasets = [load_dataset("EleutherAI/hendrycks_math", config, split="train") for config in configs]
    train_dataset = concatenate_datasets(datasets).map(_preprocess_fn)

    test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").map(_preprocess_fn)

    train_dataset = DatasetRegistry.register_dataset(TRAIN_DATASET_NAME, train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset(TEST_DATASET_NAME, test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    prepare_two_attempt_math_data()
