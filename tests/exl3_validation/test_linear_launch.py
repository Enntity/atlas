# SPDX-License-Identifier: AGPL-3.0-only
import struct
import unittest
from linear_launch import cache_record, launch_key, integer_constant

class LaunchCacheTests(unittest.TestCase):
    def test_actual_shape_key_separates_f32_output_and_padded_rows(self):
        self.assertEqual(launch_key(1,4096,77440,4,0,5,48,3),
                         launch_key(2,4096,77440,4,0,5,48,3))
        self.assertNotEqual(launch_key(1,4096,77440,4,0,5,48,3),
                            launch_key(3,4096,77440,4,0,5,48,3))
    def test_cache_last_record_owns_launch_and_corruptions_fail_closed(self):
        header=struct.pack('<8sII',b'EX3ATUNE',1,32)
        first=struct.pack('<QiiiiII',37,2,512,48,1,0,0)
        last=struct.pack('<QiiiiII',37,2,512,24,1,0,0)
        self.assertEqual(cache_record(header+first+last,37,2)['grid'],[24,1,1])
        for payload,key,tag in [(header+first,38,2),(header+first,37,3),
                                (header+first[:-1],37,2),(b'',37,2)]:
            with self.assertRaises(ValueError):cache_record(payload,key,tag)
    def test_constants_require_primary_source_and_no_conflict(self):
        self.assertEqual(integer_constant({'x':'constexpr uint64_t COOP_AUTOTUNE_VERSION = 3;'},
                                          'COOP_AUTOTUNE_VERSION'),3)
        with self.assertRaises(ValueError):integer_constant({},'CC_BLACKWELL')
        with self.assertRaises(ValueError):integer_constant({'a':'#define CC_BLACKWELL 5',
            'b':'#define CC_BLACKWELL 6'},'CC_BLACKWELL')

if __name__=='__main__':unittest.main()
