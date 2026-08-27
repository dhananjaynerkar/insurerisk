from pathlib import Path


def test_core_paths_exist():
    assert Path("src/models/train.py").exists()
    assert Path("src/models/predict.py").exists()
    assert Path("src/models/risk_service.py").exists()
    assert Path("configs/config.yaml").exists()

