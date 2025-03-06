import unittest

class AdaptivePolishTest(unittest.TestCase):

    def _setup_microscope(self):
        """Sets up a demo microscope"""
        pass

    def _setup_experiment(self):
        """Sets up a demo autolamella experiment"""
        pass

    def test_default_config(self):
        """Tests that default config works with no errors"""
        # Create AdaptivePolish obj with default input parameters from millingstrategy
        pass

    def test_ap_folders_created(self):
        """Tests that the lamella folders are created in the correct place"""
        self._setup_microscope()
        self._setup_experiment()
        pass

    def test_imaging_settings_applied(self):
        """Test that the config imaging settings are applied rather than
        existing settings if they are different"""
        pass

    def test_reference_images_saved_correctly(self):
        """Test that reference images are saved in the correct filenaming
        convention
        """
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
