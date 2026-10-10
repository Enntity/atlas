# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from linear_probe import validate_profile

class ExperimentalGeometryTests(unittest.TestCase):
    def test_head_allows_n128_and_rejects_invented_n256_tail(self):
        validate_profile([32,128,48,0],4096,77440)
        validate_profile([16,512,48,0],4096,6144)
        for profile in ([32,256,48,0],[32,512,48,0],[32,128,0,0],[32,128,48,2]):
            with self.assertRaises(ValueError):validate_profile(profile,4096,77440)

if __name__=='__main__':unittest.main()

class ProducerTransportTests(unittest.TestCase):
    def test_transport_requires_bf16_producer_identity_not_unrounded_output_h(self):
        import numpy as np
        from linear_probe import transport_comparison
        reference=np.array([[0.,1.]],np.float32)
        self.assertTrue(transport_comparison(reference,reference)['byte_exact'])
        changed=reference.copy();changed.view(np.uint32)[0,1]+=1
        self.assertFalse(transport_comparison(reference,changed)['byte_exact'])
        changed=reference.copy();changed[0,0]=-0.
        self.assertFalse(transport_comparison(reference,changed)['byte_exact'])
