import unittest

import numpy as np

from geometric.cropper import resize_crop_for_labeling


class ResizeCropForLabelingTests(unittest.TestCase):
    def test_should_return_original_crop_when_crop_is_smaller_than_target(self):
        crop = np.zeros((120, 240, 3), dtype=np.uint8)

        resized_crop = resize_crop_for_labeling(crop, 640)

        self.assertIs(resized_crop, crop)
        self.assertEqual(resized_crop.shape, (120, 240, 3))

    def test_should_downscale_crop_when_crop_exceeds_target_dimension(self):
        crop = np.zeros((600, 1200, 3), dtype=np.uint8)

        resized_crop = resize_crop_for_labeling(crop, 640)

        self.assertEqual(resized_crop.shape, (320, 640, 3))

    def test_should_preserve_aspect_ratio_when_downscaling(self):
        crop = np.zeros((900, 600, 3), dtype=np.uint8)

        resized_crop = resize_crop_for_labeling(crop, 640)

        self.assertEqual(resized_crop.shape, (640, 427, 3))


if __name__ == "__main__":
    unittest.main()
