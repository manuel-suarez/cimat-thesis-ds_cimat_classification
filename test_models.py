import os
import torch
import unittest
import itertools

from models import build_model, model_names, model_class
from torchview import draw_graph


class TestBase(unittest.TestCase):
    pass


def create_test_shape_for_encoder(model_name, encoder_name):
    def test_shape(self):
        print(f"Testing shape {model_name}-{encoder_name}")
        input_tensor = torch.randn(8, 3, 256, 256)
        model = build_model(
            model_name=model_name,
            encoder_name=encoder_name,
            in_channels=3,
            out_channels=1,
        )
        output = model(input_tensor)
        self.assertEqual((8, 1, 256, 256), output.shape)

    return test_shape


def create_test_for_encoder(model_name, encoder_name, model_type="segmentation"):
    def test_encoder(self):
        print(f"Testing graph {model_name}-{encoder_name}")
        try:
            model = build_model(
                model_name=model_name,
                encoder_name=encoder_name,
                in_channels=1,
                out_channels=1,
                model_type=model_type,
            )
            draw_graph(
                model,
                input_size=(1, 1, 256, 256),
                depth=5,
                show_shapes=True,
                expand_nested=True,
                save_graph=True,
                filename=f"encoder",
                directory=os.path.join("figures", model_name, encoder_name),
            )
            return
        except Exception as e:
            self.fail(
                f"No se pudo crear el modelo: {model_name}, encoder: {encoder_name}, excepción: ({e})"
            )

    return test_encoder


encoder_names = ["resnet", "senet", "cbamnet"]
encoder_sizes = [18, 34, 50, 101, 152]
encoders = [
    "base",
    *[
        name + str(size)
        for name, size in itertools.product(encoder_names, encoder_sizes)
    ],
]

for model_name in model_names():
    tests = {
        f"test_{encoder}_encoder": create_test_for_encoder(model_name, encoder)
        for encoder in encoders
    }
    test_shapes = {
        f"test_{encoder}_shape": create_test_shape_for_encoder(model_name, encoder)
        for encoder in encoders
    }
    tests.update(test_shapes)
    model_class_name = model_class(model_name).__name__
    classname = f"Test{model_class_name}"
    globals()[classname] = type(classname, (TestBase,), tests)
    # Model test for classification
    model_name = "classification"
    model_class_name = "ClassificationModel"
    classname = f"Test{model_class_name}"
    tests = {
        f"test_{encoder}_encoder": create_test_for_encoder(
            model_name, encoder, model_type="classification"
        )
        for encoder in encoders
    }
    globals()[classname] = type(classname, (TestBase,), tests)


if __name__ == "__main__":
    print(encoders)
