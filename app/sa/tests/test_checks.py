"""System check for the shipped model files."""

from __future__ import annotations

from unittest import mock

from django.core.checks import run_checks
from django.test import SimpleTestCase

from sa.checks import model_files_present
from sa.core import predictor


class ModelFilesCheckTests(SimpleTestCase):
    def test_passes_on_a_complete_checkout(self) -> None:
        self.assertEqual(model_files_present(None), [])

    def test_is_registered(self) -> None:
        missing = predictor.catchment_classifier_path + ".absent"
        with mock.patch.object(predictor, "catchment_classifier_path", missing):
            self.assertIn("sa.E001", [message.id for message in run_checks()])

    def test_reports_a_missing_model_run(self) -> None:
        paths = (*predictor.gnn_model_paths[:-1], predictor.gnn_model_paths[-1] + ".absent")
        with mock.patch.object(predictor, "gnn_model_paths", paths):
            errors = model_files_present(None)

        self.assertEqual([error.id for error in errors], ["sa.E001"])
        self.assertIn(paths[-1], errors[0].msg)

    def test_skips_the_recommendation_model_when_it_is_switched_off(self) -> None:
        paths = (predictor.gnn_model_paths[0] + ".absent",)
        with (
            mock.patch.object(predictor, "gnn_model_paths", paths),
            mock.patch.dict("os.environ", {"GNN_ENABLED": "false"}),
        ):
            self.assertEqual(model_files_present(None), [])
