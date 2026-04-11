from __future__ import annotations

import hydra

from projects.two_attempt_math.agent import CompactStateSelfCorrectionAgent
from projects.two_attempt_math.env import TwoAttemptSelfCorrectionEnv
from projects.two_attempt_math.prepare_data import TEST_DATASET_NAME, TRAIN_DATASET_NAME, prepare_two_attempt_math_data
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(TEST_DATASET_NAME, "test")

    if train_dataset is None or val_dataset is None:
        train_dataset, val_dataset = prepare_two_attempt_math_data()

    trainer = AgentTrainer(
        agent_class=CompactStateSelfCorrectionAgent,
        env_class=TwoAttemptSelfCorrectionEnv,
        agent_args={},
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
