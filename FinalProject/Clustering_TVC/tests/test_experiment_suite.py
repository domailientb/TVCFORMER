import math

from FinalProject.Clustering_TVC.run_experiment_suite import valid_completed_row


def _row(**updates):
    row = {
        "status": "completed", "validation_mse": "0.6", "validation_mae": "0.5",
        "test_mse": "0.3", "test_mae": "0.4", "best_epoch": "5",
        "training_time": "30", "test_samples": "2785", "total_time": "35",
        "pred_len": "96", "seed": "2021",
    }
    row.update(updates)
    return row


def test_resume_validation_accepts_valid_low_mse_and_completion_marker(tmp_path):
    log = tmp_path / "seed.log"
    log.write_text("HORIZON_COMPLETED pred_len=96 seed=2021 test_samples=2785\n")
    assert valid_completed_row(_row(), log)


def test_resume_validation_rejects_missing_marker_or_nonfinite_metric(tmp_path):
    log = tmp_path / "seed.log"
    log.write_text("run ended\n")
    assert not valid_completed_row(_row(), log)
    log.write_text("HORIZON_COMPLETED pred_len=96 seed=2021 test_samples=2785\n")
    assert not valid_completed_row(_row(validation_mae=str(math.nan)), log)


def test_resume_validation_rejects_zero_test_samples(tmp_path):
    log = tmp_path / "seed.log"
    log.write_text("HORIZON_COMPLETED pred_len=96 seed=2021 test_samples=0\n")
    assert not valid_completed_row(_row(test_samples="0"), log)
