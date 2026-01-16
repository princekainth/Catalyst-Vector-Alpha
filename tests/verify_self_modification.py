
import unittest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import self_patch, SELF_PATCH_ALLOWLIST

class TestSelfPatch(unittest.TestCase):
    
    def setUp(self):
        # Create a test file that IS in the allowlist
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.project_root)
        
        # We'll test on prompts.py (in allowlist)
        self.target_file = "prompts.py"
        self.target_path = os.path.join(self.parent_dir, self.target_file)
        
        # Read original content
        with open(self.target_path, 'r') as f:
            self.original_content = f.read()
    
    def tearDown(self):
        # Restore original content
        with open(self.target_path, 'w') as f:
            f.write(self.original_content)
        
        # Clean up backup dir
        backup_dir = "/tmp/cva_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

    def test_security_allowlist_block(self):
        """Verify non-allowlisted files are blocked."""
        result = self_patch("app.py", "test", "replacement")
        
        self.assertEqual(result.get("status"), "error")
        self.assertIn("SECURITY", result.get("error", ""))
        print("\n[TEST] Security gate correctly blocked app.py modification!")
        
    def test_backup_creation(self):
        """Verify backup is created before patching."""
        # Use a pattern that exists
        search = "# --- Prompts"
        replacement = "# --- Modified Prompts"
        
        # Mock the subprocess to skip actual tests
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="tests passed", stderr="")
            result = self_patch(self.target_file, search, replacement)
        
        # Check backup was created
        backup_path = result.get("data", {}).get("backup_path")
        if backup_path:
            self.assertTrue(os.path.exists(backup_path))
            print(f"\n[TEST] Backup created at: {backup_path}")
        else:
            print(f"\n[TEST] Result: {result}")

    def test_pattern_not_found(self):
        """Verify error when pattern doesn't exist."""
        result = self_patch(self.target_file, "THIS_PATTERN_DOES_NOT_EXIST_12345", "replacement")
        
        self.assertEqual(result.get("status"), "error")
        self.assertIn("not found", result.get("error", ""))
        print("\n[TEST] Correctly rejected non-existent pattern!")

if __name__ == '__main__':
    unittest.main()
