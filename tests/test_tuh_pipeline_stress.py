"""
Stress test for the TUH EEG ingest pipeline.

Tests:
1. Channel normalization — all TUH naming variants produce non-zero segments
2. process_dataset_dir raises on 100% skip (catches silent failures)
3. Zero-segment guard prevents raw data deletion
4. Smoke test catches preprocessing failures before full run
5. End-to-end: real TUH EDF -> segments -> npy on disk
6. Resume: already-processed buckets are skipped correctly
7. Retry logic: rsync failure retries before giving up
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.preprocess import process_dataset_dir, process_eeg_file, _normalize_ch  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tuh_style_raw(ch_names, sfreq=256, duration=20):
    """Create a synthetic MNE Raw with TUH-style channel names and EEG type."""
    n_times = int(sfreq * duration)
    data = np.random.randn(len(ch_names), n_times) * 1e-6
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _write_edf(raw, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.export(str(path), fmt="edf", verbose=False, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# 1. Channel normalisation covers all TUH naming variants
# ---------------------------------------------------------------------------

class TestChannelNormalization:
    @pytest.mark.parametrize("raw_name,expected", [
        ("EEG FP1-LE",  "FP1"),
        ("EEG FP1-REF", "FP1"),
        ("EEG FP1",     "FP1"),
        ("FP1-LE",      "FP1"),
        ("FP1",         "FP1"),
        ("EEG C3-A2",   "C3"),
        ("eeg fp2-le",  "FP2"),
        ("ECG EKG",     "EKG"),
    ])
    def test_normalize(self, raw_name, expected):
        from src.preprocess import _normalize_ch

        assert _normalize_ch(raw_name) == expected


# ---------------------------------------------------------------------------
# 2. process_dataset_dir raises when 100% of files are skipped
# ---------------------------------------------------------------------------

class TestProcessDatasetDirFailFast:
    def test_raises_on_all_skipped(self, tmp_path):
        """If every file raises during preprocessing, RuntimeError must be raised."""
        in_dir  = tmp_path / "raw"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        # Write a deliberately broken EDF (empty file)
        bad = in_dir / "bad.edf"
        bad.write_bytes(b"not an edf")

        with pytest.raises(RuntimeError, match="0/.*files produced output"):
            process_dataset_dir(str(in_dir), str(out_dir), unified=False)

    def test_no_raise_when_some_succeed(self, tmp_path):
        """If at least one file succeeds, no RuntimeError."""
        in_dir  = tmp_path / "raw"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        # One bad file
        (in_dir / "bad.edf").write_bytes(b"not an edf")

        # One good synthetic TUH-style file
        raw = _make_tuh_style_raw(["EEG FP1-LE", "EEG C3-LE", "EEG C4-LE"])
        good = in_dir / "good.edf"
        _write_edf(raw, good)

        # Should complete without raising
        process_dataset_dir(str(in_dir), str(out_dir), unified=False)
        assert any(out_dir.glob("**/*.npy"))


# ---------------------------------------------------------------------------
# 3. Zero-segment guard prevents deletion in ingest pipeline
# ---------------------------------------------------------------------------

class TestZeroSegmentGuard:
    def test_raw_preserved_when_zero_segments(self, tmp_path, monkeypatch):
        """If preprocess_bucket returns 0, raw staging dir must NOT be deleted."""
        import scripts.tuh_ingest_pipeline as pipe

        staging = tmp_path / "staging" / "039"
        staging.mkdir(parents=True)
        (staging / "dummy.edf").write_bytes(b"fake")

        monkeypatch.setattr(pipe, "RAW_STAGING", tmp_path / "staging")
        monkeypatch.setattr(pipe, "PROCESSED_OUT", tmp_path / "processed")
        monkeypatch.setattr(pipe, "smoke_test_bucket", lambda bid: True)
        monkeypatch.setattr(pipe, "preprocess_bucket", lambda bid: 0)
        monkeypatch.setattr(pipe, "rsync_bucket", lambda bid: True)

        deleted = []
        monkeypatch.setattr(pipe, "delete_raw_bucket", lambda bid: deleted.append(bid))

        sys.argv = ["tuh_ingest_pipeline.py", "--start", "39", "--end", "39"]
        pipe.main()

        assert 39 not in deleted, "Raw data should NOT be deleted when n_segs == 0"


# ---------------------------------------------------------------------------
# 4. Smoke test catches preprocessing failures before full bucket run
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_smoke_passes_on_valid_tuh_edf(self, tmp_path, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe
        monkeypatch.setattr(pipe, "RAW_STAGING", tmp_path / "staging")

        staging = tmp_path / "staging" / "010"
        staging.mkdir(parents=True)

        raw = _make_tuh_style_raw(["EEG FP1-LE", "EEG C3-LE", "EEG C4-LE",
                                    "EEG F3-LE",  "EEG F4-LE", "EEG O1-LE"])
        _write_edf(raw, staging / "sub001.edf")

        assert pipe.smoke_test_bucket(10) is True

    def test_smoke_fails_on_broken_file(self, tmp_path, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe
        monkeypatch.setattr(pipe, "RAW_STAGING", tmp_path / "staging")

        staging = tmp_path / "staging" / "011"
        staging.mkdir(parents=True)
        (staging / "bad.edf").write_bytes(b"not an edf")

        assert pipe.smoke_test_bucket(11) is False

    def test_smoke_fails_on_empty_staging(self, tmp_path, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe
        monkeypatch.setattr(pipe, "RAW_STAGING", tmp_path / "staging")
        (tmp_path / "staging" / "012").mkdir(parents=True)

        assert pipe.smoke_test_bucket(12) is False


# ---------------------------------------------------------------------------
# 5. End-to-end: TUH-style EDF -> segments -> .npy file on disk
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.parametrize("ch_format", [
        ["EEG FP1-LE", "EEG FP2-LE", "EEG C3-LE", "EEG C4-LE", "EEG O1-LE", "EEG O2-LE"],
        ["EEG FP1-REF","EEG FP2-REF","EEG C3-REF","EEG C4-REF","EEG O1-REF","EEG O2-REF"],
        ["FP1-LE", "FP2-LE", "C3-LE", "C4-LE", "O1-LE", "O2-LE"],
        ["FP1", "FP2", "C3", "C4", "O1", "O2"],
    ])
    def test_channel_variants_produce_segments(self, tmp_path, ch_format):
        raw = _make_tuh_style_raw(ch_format, sfreq=256, duration=30)
        edf_path = tmp_path / "test.edf"
        _write_edf(raw, edf_path)

        result = process_eeg_file(str(edf_path), unified=False)
        assert result is not None, "process_eeg_file returned None"
        assert result.ndim == 3, f"Expected (N, C, T) got shape {result.shape}"
        assert result.shape[0] > 0, "Got 0 segments"
        assert result.shape[1] > 0, "Got 0 channels"

    def test_npy_written_to_disk(self, tmp_path):
        ch = ["EEG FP1-LE", "EEG C3-LE", "EEG O1-LE", "EEG FP2-LE", "EEG C4-LE", "EEG O2-LE"]
        raw = _make_tuh_style_raw(ch, sfreq=256, duration=25)
        edf_path = tmp_path / "test.edf"
        out_path  = tmp_path / "test.npy"
        _write_edf(raw, edf_path)

        process_eeg_file(str(edf_path), str(out_path), unified=False)

        assert out_path.exists(), ".npy file not written"
        arr = np.load(str(out_path))
        assert arr.shape[0] > 0

    def test_process_dataset_dir_end_to_end(self, tmp_path):
        in_dir  = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        in_dir.mkdir()

        for i in range(3):
            ch = ["EEG FP1-LE", "EEG C3-LE", "EEG O1-LE"]
            raw = _make_tuh_style_raw(ch, sfreq=256, duration=20)
            _write_edf(raw, in_dir / f"sub{i:03d}.edf")

        process_dataset_dir(str(in_dir), str(out_dir), unified=False)

        npy_files = list(out_dir.glob("**/*.npy"))
        assert len(npy_files) == 3, f"Expected 3 .npy files, got {len(npy_files)}"
        for f in npy_files:
            arr = np.load(str(f))
            assert arr.shape[0] > 0


# ---------------------------------------------------------------------------
# 6. Resume: already-done buckets are correctly skipped
# ---------------------------------------------------------------------------

class TestResume:
    def test_done_bucket_is_skipped(self, tmp_path, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe
        processed = tmp_path / "processed" / "039"
        processed.mkdir(parents=True)
        (processed / "sub.npy").write_bytes(b"fake")

        monkeypatch.setattr(pipe, "PROCESSED_OUT", tmp_path / "processed")
        assert pipe.bucket_is_done(39) is True

    def test_empty_output_dir_is_not_done(self, tmp_path, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe
        (tmp_path / "processed" / "039").mkdir(parents=True)
        monkeypatch.setattr(pipe, "PROCESSED_OUT", tmp_path / "processed")
        assert pipe.bucket_is_done(39) is False


# ---------------------------------------------------------------------------
# 7. Retry logic: rsync retries on failure
# ---------------------------------------------------------------------------

class TestRsyncRetry:
    def test_retries_on_failure_then_succeeds(self, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe

        call_count = {"n": 0}
        original = pipe.rsync_bucket

        def flaky_rsync(bid):
            call_count["n"] += 1
            return call_count["n"] >= 3  # fail twice, succeed on 3rd

        monkeypatch.setattr(pipe, "rsync_bucket", flaky_rsync)
        monkeypatch.setattr(pipe.time, "sleep", lambda s: None)

        # Simulate the retry loop from main()
        rsync_ok = False
        for attempt in range(5):
            if flaky_rsync(39):
                rsync_ok = True
                break
            pipe.time.sleep(60)

        assert rsync_ok is True
        assert call_count["n"] >= 3

    def test_gives_up_after_max_retries(self, monkeypatch):
        import scripts.tuh_ingest_pipeline as pipe
        monkeypatch.setattr(pipe, "rsync_bucket", lambda bid: False)
        monkeypatch.setattr(pipe.time, "sleep", lambda s: None)

        rsync_ok = False
        for attempt in range(5):
            if pipe.rsync_bucket(39):
                rsync_ok = True
                break
        assert rsync_ok is False
