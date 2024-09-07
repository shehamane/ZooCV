from model.model import ClassificationModel


class VGG16(ClassificationModel):
    def __init__(self, nc):
        super().__init__('../config/vgg-16.yaml', nc)

class VGG19(ClassificationModel):
    def __init__(self, nc):
        super().__init__('../config/vgg-19.yaml', nc)

