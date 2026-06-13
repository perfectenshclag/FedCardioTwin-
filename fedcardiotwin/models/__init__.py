from .inception1d import Inception1d
from .resnet1d import ResNet1d
from .adapters import PatientTwin


def build_model(name, num_classes, **kw):
    if name == "inception1d":
        return Inception1d(num_classes=num_classes, **kw)
    if name == "resnet1d":
        return ResNet1d(num_classes=num_classes, **kw)
    raise ValueError(name)
