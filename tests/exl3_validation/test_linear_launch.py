# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import struct
import unittest
from linear_launch import cache_manifest, cache_record, launch_key, integer_constant

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

class CacheManifestTests(unittest.TestCase):
    HEADER=struct.pack('<8sII',b'EX3ATUNE',1,32)
    def canonical_map_sha(self,records):
        final={record[0]:record for record in records}  # last wins in file order
        canonical=b''.join(struct.pack('<QiiiiII',*final[key]) for key in sorted(final))
        return hashlib.sha256(canonical).hexdigest()
    def test_duplicate_hash_overwrites_in_file_order_and_map_hash_matches(self):
        first=struct.pack('<QiiiiII',37,2,512,48,1,0,0)
        last=struct.pack('<QiiiiII',37,2,512,24,1,0,0)
        other=struct.pack('<QiiiiII',11,3,256,24,2,0,0)
        payload=self.HEADER+first+other+last
        manifest=cache_manifest(payload)
        records=[struct.unpack_from('<QiiiiII',payload,offset)
                 for offset in range(16,len(payload),32)]
        self.assertEqual(manifest['record_map_sha256'],self.canonical_map_sha(records))
        self.assertEqual(manifest['record_count'],3)
        self.assertEqual(manifest['unique_hash_count'],2)
        self.assertEqual(manifest['duplicate_overwrite_count'],1)
        self.assertEqual(manifest['raw_sha256'],hashlib.sha256(payload).hexdigest())
    def test_same_final_map_in_different_file_order_shares_map_hash_not_raw(self):
        h1a=struct.pack('<QiiiiII',1,2,512,48,1,0,0)
        h1b=struct.pack('<QiiiiII',1,2,512,24,1,0,0)
        h2=struct.pack('<QiiiiII',2,3,256,24,2,0,0)
        a=cache_manifest(self.HEADER+h1a+h2+h1b)
        b=cache_manifest(self.HEADER+h2+h1a+h1b)
        self.assertNotEqual(a['raw_sha256'],b['raw_sha256'])
        self.assertEqual(a['record_map_sha256'],b['record_map_sha256'])
        self.assertEqual(a['record_map_sha256'],
                         self.canonical_map_sha([struct.unpack('<QiiiiII',h1a),
                                                 struct.unpack('<QiiiiII',h2),
                                                 struct.unpack('<QiiiiII',h1b)]))
        for manifest in (a,b):
            self.assertEqual(manifest['record_count'],3)
            self.assertEqual(manifest['unique_hash_count'],2)
            self.assertEqual(manifest['duplicate_overwrite_count'],1)
    def test_header_only_manifest_is_empty_but_valid(self):
        manifest=cache_manifest(self.HEADER)
        self.assertEqual((manifest['record_count'],manifest['unique_hash_count'],
                          manifest['duplicate_overwrite_count']),(0,0,0))
        self.assertEqual(manifest['record_map_sha256'],hashlib.sha256(b'').hexdigest())
    def test_truncated_version_size_magic_and_empty_payload_rejected(self):
        record=struct.pack('<QiiiiII',37,2,512,48,1,0,0)
        cases=[b'',                                            # empty payload
               b'EX3ATU',                                      # truncated header
               record,                                         # header bytes absent: bad magic
               struct.pack('<8sII',b'EX3ATUN3',1,32)+record,   # bad magic
               struct.pack('<8sII',b'EX3ATUNE',2,32)+record,   # unknown version
               struct.pack('<8sII',b'EX3ATUNE',1,24)+record,   # wrong record size
               self.HEADER+record[:-1],                        # truncated record
               self.HEADER+record+record[:8]]                  # partial trailing record
        for payload in cases:
            with self.subTest(size=len(payload)),self.assertRaises(ValueError):
                cache_manifest(payload)

if __name__=='__main__':unittest.main()
