# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from mhc_oracle import native_mhc_compiler_options, qualification_pass, rounded_split_count


class MhcCompilePolicyTests(unittest.TestCase):
    def test_native_policy_is_explicit_and_distinct(self):
        self.assertEqual(native_mhc_compiler_options('true'),('--std=c++17','--fmad=true'))
        self.assertEqual(native_mhc_compiler_options('false'),('--std=c++17','--fmad=false'))

    def test_rounded_profile_derives_explicit_splits(self):
        self.assertEqual(rounded_split_count(10,16384,48),48)
        self.assertEqual(rounded_split_count(17,16384,48),48)
        self.assertEqual(rounded_split_count(128,16384,48),24)
        for args in ((0,16384,48),(10,16385,48),(10,16384,0)):
            with self.assertRaises(ValueError):rounded_split_count(*args)

    def test_every_corrected_tensor_is_required(self):
        replay={'normalized':{'byte_exact':True}}
        corrected={'normalized':{'byte_exact':True},'post':{'byte_exact':False}}
        self.assertFalse(qualification_pass(replay,corrected))
        corrected['post']['byte_exact']=True
        self.assertTrue(qualification_pass(replay,corrected))
        replay['normalized']['byte_exact']=False
        self.assertFalse(qualification_pass(replay,corrected))

    def test_absent_or_invalid_policy_rejected(self):
        for value in (None,'',True,'yes'):
            with self.assertRaises(ValueError):native_mhc_compiler_options(value)


if __name__=='__main__':unittest.main()
