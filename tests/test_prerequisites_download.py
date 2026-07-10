import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


try:
    import requests
except ModuleNotFoundError:
    requests = types.ModuleType("requests")

    class HTTPError(Exception):
        pass

    requests.HTTPError = HTTPError
    requests.get = None
    requests.head = None
    sys.modules["requests"] = requests

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda *args, **kwargs: None
    sys.modules["tqdm"] = tqdm_module

from rvc.lib.tools import prerequisites_download as downloader


class ProgressBar:
    def __init__(self):
        self.total = 0

    def update(self, size):
        self.total += size


class PrerequisitesDownloadTests(unittest.TestCase):
    def test_cli_does_not_report_success_after_a_download_failure(self):
        core_source = (Path(__file__).parents[1] / "core.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("traceback.print_exc()\n        raise SystemExit(1)", core_source)

    def test_download_file_writes_verified_response_atomically(self):
        response = Mock()
        response.headers = {"content-length": "6"}
        response.iter_content.return_value = [b"abc", b"", b"def"]

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "models" / "model.pth"
            progress = ProgressBar()
            with patch.object(downloader.requests, "get", return_value=response) as get:
                downloader.download_file(
                    "https://example.test/model.pth", str(destination), progress
                )

            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertFalse(Path(f"{destination}.part").exists())
            self.assertEqual(progress.total, 6)
            get.assert_called_once_with(
                "https://example.test/model.pth",
                stream=True,
                timeout=downloader.REQUEST_TIMEOUT,
            )
            response.raise_for_status.assert_called_once()

    def test_download_file_removes_partial_file_after_failure(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("not found")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.pth"
            with patch.object(downloader.requests, "get", return_value=response):
                with self.assertRaises(requests.HTTPError):
                    downloader.download_file(
                        "https://example.test/missing.pth", str(destination), ProgressBar()
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(Path(f"{destination}.part").exists())

    def test_incomplete_download_is_not_promoted_to_model_file(self):
        response = Mock()
        response.headers = {"content-length": "10"}
        response.iter_content.return_value = [b"short"]

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.pth"
            with patch.object(downloader.requests, "get", return_value=response):
                with self.assertRaisesRegex(IOError, "Incomplete download"):
                    downloader.download_file(
                        "https://example.test/model.pth", str(destination), ProgressBar()
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(Path(f"{destination}.part").exists())

    def test_empty_existing_file_is_not_treated_as_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                Path("model.pth").touch()
                with self.assertRaises(FileNotFoundError):
                    downloader.verify_downloads([("", ["model.pth"])])

                Path("model.pth").write_bytes(b"model")
                downloader.verify_downloads([("", ["model.pth"])])
            finally:
                os.chdir(previous_directory)

    def test_size_probe_follows_redirects_and_checks_status(self):
        response = Mock()
        response.headers = {"content-length": "123"}

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(
                    downloader.requests, "head", return_value=response
                ) as head:
                    size = downloader.get_file_size_if_missing([("", ["model.pth"])])
            finally:
                os.chdir(previous_directory)

        self.assertEqual(size, 123)
        head.assert_called_once_with(
            "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/model.pth",
            allow_redirects=True,
            timeout=downloader.REQUEST_TIMEOUT,
        )
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
