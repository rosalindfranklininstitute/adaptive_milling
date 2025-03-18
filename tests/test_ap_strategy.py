import unittest
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import tempfile
from datetime import datetime

import fibsem.utils

from autolamella.structures import Experiment

from adaptive_polish.strategy import AdaptivePolishMillingConfig, AdaptivePolishMillingStrategy

class AdaptivePolishTest(unittest.TestCase):

    def _setup_microscope(
            self,
            ap_config: AdaptivePolishMillingConfig,
            tmp_dir: Path = None,
        ):
        """Sets up a demo microscope"""

        # set up protocol.yaml
        if tmp_dir is None:
            tmp = tempfile.TemporaryDirectory()
            tmp_dir = tmp.name
        self.tmp_dir = tmp_dir
        environment = Environment(loader=FileSystemLoader("data/"))
        template = environment.get_template("protocol.yaml")
        protocol = template.render(ap_config.to_dict())
        with open(f"{tmp_dir}/protocol.yaml", "w") as p:
            p.write(protocol)
        self.microscope, self.settings = fibsem.utils.setup_session(
            protocol_path=f"{self.tmp_dir}/protocol.yaml"
        )

    def _setup_experiment(
            self,
            tmp_dir: Path = None,
            model_path: Path = None,
        ):
        """Sets up a demo autolamella experiment"""
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        # set up experiment.yaml
        if tmp_dir is None:
            tmp = tempfile.TemporaryDirectory()
            tmp_dir = tmp.name
        self.tmp_dir = tmp_dir
        environment = Environment(loader=FileSystemLoader("data/"))
        template = environment.get_template("experiment.yaml")
        experiment = template.render({
            "now": now,
            "cwd": str(Path.cwd()),
            "model_path": model_path,
        })
        with open(f"{tmp_dir}/experiment.yaml", "w") as p:
            p.write(experiment)

        self.exp = Experiment.load(f"{tmp_dir}/experiment.yaml")

    def test_default_config(self):
        """Tests that default config works with no errors"""
        # Create AdaptivePolish obj with default input parameters from millingstrategy
        pass

    def test_ap_folders_created(self):
        """Tests that the lamella folders are created in the correct place"""
        # self._setup_microscope()
        self._setup_experiment()
        print(self.exp)

    def test_imaging_settings_applied(self):
        """Test that the config imaging settings are applied rather than
        existing settings if they are different"""
        pass

    def test_reference_images_saved_correctly(self):
        """Test that reference images are saved in the correct filenaming
        convention
        """
        pass

    def test_segmentation_model(self):
        """Test that the segmentation model produces a prediction"""
        pass

    def test_milling_stops_when_stop_criterion_reached(self):
        """Tests that milling stops when either GIS is too thin or crack
        is found"""
        pass

    def test_max_milling_cycles_not_exceeded(self):
        """Tests that milling cycles cannot exceed the max"""
        pass

    def test_results_saved(self):
        """Tests that results are saved in correct filenaming convention"""
        pass

    def test_milling_time_adjustment(self):
        """Tests that milling cycle time cannot be < 10s"""
        pass
