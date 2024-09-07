from model.model import ClassificationModel


class LeNet5(ClassificationModel):
    def __init__(self, nc):
        super().__init__('../config/alexnet.yaml', nc)

