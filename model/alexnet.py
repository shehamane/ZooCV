from model.model import ClassificationModel


class AlexNet(ClassificationModel):
    def __init__(self, nc):
        super().__init__('../config/alexnet.yaml', nc)
