"""Synthetic numerical regression tests. No field coordinates or mineral claims."""
import unittest
import numpy as np
from acquisition.bouh_math import calibration, compute_indices, safe_divide

class ScientificIndicesTests(unittest.TestCase):
    def test_explicit_radiometric_calibration(self):
        self.assertEqual(calibration({'raster:bands':[{'scale':.0001,'offset':-.1}]}),(.0001,-.1))
        for bad in ({}, {'raster:bands':[{'scale':.0001}]},
                    {'raster:bands':[{'scale':0,'offset':0}]},
                    {'raster:bands':[{'scale':float('nan'),'offset':0}]},
                    {'raster:bands':[{'scale':True,'offset':0}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calibration(bad)

    def test_safe_divide_preserves_nodata_as_nan_not_zero(self):
        num=np.array([2.,2.,np.nan,2.],dtype='float32')
        den=np.array([1.,0.,1.,np.inf],dtype='float32')
        out=safe_divide(num,den,np)
        self.assertEqual(out.dtype,np.float32)
        self.assertEqual(float(out[0]),2.)
        self.assertTrue(np.isnan(out[1:]).all())
        self.assertFalse(np.isinf(out).any())

    def test_bare_pixel_masks_and_per_ratio_missingness(self):
        arrays={
          'blue':np.array([[.2,.2],[.2,0.]],dtype='float32'),
          'green':np.array([[.15,.2],[.15,.05]],dtype='float32'),
          'red':np.array([[.3,.1],[np.nan,.1]],dtype='float32'),
          'nir':np.array([[.25,.4],[.25,.11]],dtype='float32'),
          'swir16':np.array([[.5,.2],[.5,.4]],dtype='float32'),
          'swir22':np.array([[.25,.2],[.25,.2]],dtype='float32')}
        out=compute_indices(arrays,np.ones((2,2),dtype=bool),np)
        self.assertEqual(out['valid_bare_mask'].tolist(),[[1,0],[0,1]])
        self.assertAlmostEqual(float(out['iron_ratio'][0,0]),1.5,places=5)
        self.assertAlmostEqual(float(out['swir_ratio'][0,0]),2.,places=5)
        self.assertTrue(np.isnan(out['iron_ratio'][0,1]))
        self.assertTrue(np.isnan(out['iron_ratio'][1,0]))
        self.assertTrue(np.isnan(out['iron_ratio'][1,1]))
        self.assertAlmostEqual(float(out['swir_ratio'][1,1]),2.,places=5)
        self.assertNotIn('gold_probability',out)

    def test_bad_grid_and_non_boolean_mask_fail_closed(self):
        arrays={key:np.ones((2,2),dtype='float32') for key in
            ('blue','green','red','nir','swir16','swir22')}
        with self.assertRaisesRegex(ValueError,'MISMATCHED_GRIDS_OR_MASK_TYPE'):
            compute_indices(arrays,np.ones((2,2),dtype=np.uint8),np)
        arrays['red']=np.ones((1,2),dtype='float32')
        with self.assertRaisesRegex(ValueError,'MISMATCHED_GRIDS_OR_MASK_TYPE'):
            compute_indices(arrays,np.ones((2,2),dtype=bool),np)

    def test_masked_out_pixel_has_no_ratio(self):
        arrays={key:np.full((2,2),.1,dtype='float32') for key in
            ('blue','green','red','nir','swir16','swir22')}
        mask=np.array([[True,False],[False,True]])
        out=compute_indices(arrays,mask,np)
        for field in ('iron_ratio','swir_ratio','gossan_ratio'):
            self.assertTrue(np.isnan(out[field][0,1]))
            self.assertTrue(np.isnan(out[field][1,0]))

if __name__=='__main__':
    unittest.main()
