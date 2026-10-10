# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from pathlib import Path
from kda_lt_oracle import validate_pair, validate_workspace

class Tests(unittest.TestCase):
    def test_workspace_matches_existing_lt_operation(self):
        validate_workspace(64*1024*1024)
        with self.assertRaises(ValueError):validate_workspace(32*1024*1024)
    def test_exact_pair_rejects_foreign_provider(self):
        pair=[Path('/candidate/classic'),Path('/candidate/lt')]
        validate_pair(set(pair),pair)
        with self.assertRaises(RuntimeError):validate_pair({pair[1]},pair)
        with self.assertRaises(RuntimeError):validate_pair(set(pair)|{Path('/reference/lt')},pair)

if __name__=='__main__':unittest.main()
