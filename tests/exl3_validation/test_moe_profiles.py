# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from moe_precision import validate_mma_profiles


class ProfileTests(unittest.TestCase):
    def test_explicit_complete_profiles(self):
        validate_mma_profiles(object(), (32, 256, 8, 1), (16, 128, 1, 0))
        validate_mma_profiles(None, None, None)

    def test_no_implicit_fat_policy(self):
        with self.assertRaises(ValueError):
            validate_mma_profiles(object(), (32, 256, 8, 1), None)

    def test_wrong_fat_and_thin_contracts_rejected(self):
        for thin, fat in (((32,256,8,1),(32,128,1,0)),
                          ((32,256,8,0),(16,128,1,0))):
            with self.assertRaises(ValueError):
                validate_mma_profiles(object(), thin, fat)
