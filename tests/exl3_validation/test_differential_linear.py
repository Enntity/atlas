# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from differential_linear import bit_comparison, reference_rows, guarded_payload, selected_geometry, validate_mul1_marker, output_component_bounds, require_aligned_pointers, fused_reference_layout

class LinearBoundaryTests(unittest.TestCase):
    def test_reference_executes_whole_local_fused_operator_before_component_slice(self):
        self.assertEqual(fused_reference_layout([8192]*3,1,4096),(12288,4096))
        self.assertEqual(fused_reference_layout([8192]*3,2,4096),(12288,8192))
        for args in [([8192,4096,8192],0,4096),([8192]*3,0,12288)]:
            with self.assertRaises(ValueError):fused_reference_layout(*args)
    def test_guarded_payload_pointer_must_retain_packed_vector_alignment(self):
        require_aligned_pointers([256,256+32*4,256+64*2])
        for pointers in [[256+4],[256+2],[0],[]]:
            with self.assertRaises(ValueError):require_aligned_pointers(pointers)
    def test_fused_output_component_selects_each_rank_within_component(self):
        self.assertEqual(output_component_bounds([512,256,512],1,0),(512,128))
        self.assertEqual(output_component_bounds([512,256,512],1,1),(640,128))
        self.assertEqual(output_component_bounds([512,256,512],2,1),(1024,256))
        for args in [([128,256],0,0),([512,0],0,0),([512],1,0),([],0,0),([512],0,2)]:
            with self.assertRaises(ValueError):output_component_bounds(*args)
    def test_mul1_is_signed_i32_codebook_marker_rather_than_boolean_one(self):
        validate_mul1_marker(np.array(-2082680531,dtype='<i4'),0x83dcd12d)
        for values in [np.array(1,dtype='<i4'),np.array(-2082680531,dtype='<i8'),
                       np.array([-2082680531,-2082680531],dtype='<i4')]:
            with self.assertRaises(ValueError):validate_mul1_marker(values,0x83dcd12d)
    def test_runtime_tag_geometry_comes_from_bound_source_and_rejects_ambiguity(self):
        source='#define EXL3_GEMM_SHAPE_2 16, 32, 128, 4, 3\n'
        self.assertEqual(selected_geometry(2,{'map.cuh':source})['tile_n'],128)
        self.assertEqual(selected_geometry(90,{})['path'],'qtip_gemv')
        with self.assertRaises(ValueError):selected_geometry(3,{'map.cuh':source})
        with self.assertRaises(ValueError):selected_geometry(2,{'a':source,'b':source})
    def test_exact_comparison_detects_signed_zero_and_single_ulp(self):
        x=np.array([[0.,1.]],np.float32)
        self.assertEqual(bit_comparison(x,x.copy())['bit_mismatches'],0)
        y=x.copy();y[0,0]=-0.
        self.assertEqual(bit_comparison(x,y)['bit_mismatches'],1)
        y=x.copy();y.view(np.uint32)[0,1]+=1
        self.assertEqual(bit_comparison(x,y)['bit_mismatches'],1)
        with self.assertRaises(ValueError):bit_comparison(x,np.full_like(x,np.inf))
    def test_reference_padding_is_explicit_and_head_does_not_pad(self):
        self.assertEqual(reference_rows(1,3,False),1)
        self.assertEqual(reference_rows(1,3,True),3)
        self.assertEqual(reference_rows(5,3,True),5)
        for args in [(0,3,True),(1,0,True),(129,3,True)]:
            with self.assertRaises(ValueError):reference_rows(*args)
    def test_guard_failure_detected_separately_from_output_error(self):
        aligned=np.full(67,99,np.float32);aligned[32:35]=[1,2,3]
        np.testing.assert_array_equal(guarded_payload(aligned,99,(1,3),32),[[1,2,3]])
        aligned[15]=98
        with self.assertRaises(ValueError):guarded_payload(aligned,99,(1,3),32)
        x=np.array([99,1,2,3,99],np.uint16)
        np.testing.assert_array_equal(guarded_payload(x,99,(1,3),1),[[1,2,3]])
        x[-1]=98
        with self.assertRaises(ValueError):guarded_payload(x,99,(1,3),1)
if __name__=='__main__':unittest.main()
