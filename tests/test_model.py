import pytest

def test_model_output_shape():
    pytest.importorskip("tensorflow")
    from models.model import build_model

    model= build_model()
    assert model.input_shape== (None, 32, 32, 3)
    assert model.output_shape== (None, 10)