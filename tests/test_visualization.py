# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import pytest

import rflib


def test_color():
    assert rflib.color_val(rflib.Color.blue) == (255, 0, 0)
    assert rflib.color_val('green') == (0, 255, 0)
    assert rflib.color_val((1, 2, 3)) == (1, 2, 3)
    assert rflib.color_val(100) == (100, 100, 100)
    assert rflib.color_val(np.zeros(3, dtype=np.int)) == (0, 0, 0)
    with pytest.raises(TypeError):
        rflib.color_val([255, 255, 255])
    with pytest.raises(TypeError):
        rflib.color_val(1.0)
    with pytest.raises(AssertionError):
        rflib.color_val((0, 0, 500))
