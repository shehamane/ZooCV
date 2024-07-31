import abc
import numpy as np
import cv2
import torch


class Sample:
    class SampleData:
        def __init__(self, labels=None, bboxes=None, masks=None):
            self.labels = labels
            self.bboxes = bboxes
            self.masks = masks

    def __init__(self, im, labels=None, bboxes=None, masks=None, gt_labels=None, gt_bboxes=None, gt_masks=None):
        self.im = im
        self.pred_data = Sample.SampleData(labels, bboxes, masks)
        self.gt_data = Sample.SampleData(gt_labels, gt_bboxes, gt_masks)


class SampleDrawer:
    def __init__(self, resize=None, convert=None):
        self.sample = None
        self.drawing = None
        self.resize = resize
        self.convert = convert

    def take_sample(self, sample):
        self.sample = sample
        self.drawing = cv2.resize(sample.im.copy(), self.resize, interpolation=cv2.INTER_CUBIC)
        if self.convert is not None:
            self.drawing = cv2.cvtColor(self.drawing, self.convert)

    def draw_labels(self, gt=False, pos=(-5, 5), color=(255, 0, 0), thickness=1, background=False, names=None):
        labels = self.sample.gt_data.labels if gt else self.sample.pred_data.labels

        if isinstance(labels, int):
            text = names[labels] if names else str(labels)
        elif isinstance(labels, tuple):
            text = ', '.join([names[label] for label in labels]) if names else ', '.join(labels)
        elif isinstance(labels, dict):
            text = ', '.join([f'{names[label]} : {score:>3f}' for label, score in labels.values()]) if names else \
                ', '.join([f'{label} : {score:>3f}' for label, score in labels.values()])
        else:
            raise Exception(f'Unsupported labels: {type(labels)}')

        self.drawing = cv2.putText(self.drawing, text, pos, cv2.FONT_HERSHEY_PLAIN, 1, color, thickness, cv2.LINE_AA)

    def get_drawing(self):
        return self.drawing


class Sampler(abc.ABC):

    @abc.abstractmethod
    def get_sample(self, im, label=None):
        raise NotImplementedError

    def from_iterable(self, data, n, with_labels):
        indexes = np.random.choice(len(data), n, replace=False)
        for i in indexes:
            im, label = data[i]
            yield self.get_sample(im, label if with_labels else None)

    def __call__(self, data, n, with_labels=True):
        return self.from_iterable(data, n, with_labels)


class ClassificationSampler(Sampler):
    def __init__(self, model, with_scores=False, device='cuda'):
        self.device = torch.device(device)
        self.model = model
        self.with_scores = with_scores

    def get_sample(self, im, label=None):
        im_t = im.to(self.device).unsqueeze(0)
        logits = self.model(im_t).detach().cpu().numpy()
        im = im.detach().cpu().numpy().transpose(1, 2, 0)

        if self.with_scores:
            return Sample(im, labels={cls: score for cls, score in enumerate(logits)}, gt_labels=label)

        label_pred = int(logits.argmax(axis=1))
        return Sample(im, labels=label_pred, gt_labels=label)
