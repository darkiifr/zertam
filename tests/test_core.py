import unittest
import os
import sys

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.marketplace import AVAILABLE_MODELS, get_model_status

class TestZertamCore(unittest.TestCase):
    
    def test_marketplace_structure(self):
        """Verify all models have required fields."""
        required = ["id", "name", "url_model", "filename_model"]
        for model in AVAILABLE_MODELS:
            for field in required:
                self.assertIn(field, model, f"Model {model.get('id')} missing {field}")
                
    def test_model_filenames(self):
        """Verify filenames are safe strings."""
        for model in AVAILABLE_MODELS:
            self.assertFalse(".." in model["filename_model"])
            self.assertFalse("/" in model["filename_model"])

if __name__ == "__main__":
    unittest.main()
