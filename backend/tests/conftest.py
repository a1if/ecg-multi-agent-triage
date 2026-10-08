from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RECORDS = ROOT / "data" / "records"


@pytest.fixture(scope="session")
def sender():
    from ecg_agent.core.sender import Sender

    return Sender(MODELS / "cnn_lstm_rr_seed0.pt", device="cpu")


@pytest.fixture
def policy():
    from ecg_agent.agent.policy import Policy

    return Policy.from_manifest(MODELS / "manifest.json")
