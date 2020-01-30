'''
Defines custom losses and metrics, with tests when running directly (python3 metrics.py)

Author: Diedre Carmo
https://github.com/dscarmo
'''
from sys import argv
from typing import List
import sys
import random
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from torch import Tensor, einsum
from utils import random_rectangle, ESC, simplex, one_hot, split_l_r
from transforms import ToTensor, Compose, CenterCrop
from scipy.spatial.distance import directed_hausdorff
from nathip import NatHIP


class EvalMetricCalculator():
    def __init__(self, metric):
        self.metric = metric
        self.accs, self.laccs, self.raccs, self.specs, self.precs, self.recs = [], [], [], [], [], []

    def __call__(self, preds, tgt):
        preds = preds.float()
        tgt = tgt.float()
        metrics = return_eval_metrics(self.metric, preds, tgt)
        self.acc = metrics["dice"]
        self.accs.append(metrics["dice"])
        self.laccs.append(metrics["ldice"])
        self.raccs.append(metrics["rdice"])
        self.specs.append(metrics["spec"])
        self.precs.append(metrics["prec"])
        self.recs.append(metrics["rec"])

    def final_results(self):
        npaccs = np.array(self.accs)
        mean, std = npaccs.mean(), npaccs.std()

        lnpaccs = np.array(self.laccs)
        lmean, lstd = lnpaccs.mean(), lnpaccs.std()

        rnpaccs = np.array(self.raccs)
        rmean, rstd = rnpaccs.mean(), rnpaccs.std()

        npspecs = np.array(self.specs)
        specsmean, specsstd = npspecs.mean(), npspecs.std()

        nprec = np.array(self.recs)
        recmean, recstd = nprec.mean(), nprec.std()

        npprec = np.array(self.precs)
        precmean, precstd = npprec.mean(), npprec.std()

        result = ''
        result += "Both Dice Mean: {}, std: {}\n".format(mean, std)
        result += "Left Dice mean: {}, std: {}\n".format(lmean, lstd)
        result += "Right Dice mean: {}, std: {}\n".format(rmean, rstd)
        result += "Specificity mean: {}, std: {}\n".format(specsmean, specsstd)
        result += "Recall mean: {}, std: {}\n".format(recmean, recstd)
        result += "Precision mean: {}, std: {}\n".format(precmean, precstd)
        print(result)
        return result


def return_eval_metrics(metric, preds, tgt):
    splitted_preds = split_l_r(preds)
    splitted_tgt = split_l_r(tgt)

    dice = metric(preds, tgt)
    ldice = metric(splitted_preds["left"], splitted_tgt["left"])
    rdice = metric(splitted_preds["right"], splitted_tgt["right"])
    spec = specificity(preds, tgt).item()
    rec = recall(preds, tgt).item()
    prec = precision(preds, tgt).item()

    return {"dice": dice, "ldice": ldice, "rdice": rdice, "spec": spec, "prec": prec, "rec": rec}


def precision(pred, tgt):
    '''
    True positives / (true positives + false positives)
    '''
    assert pred.shape == tgt.shape
    if isinstance(pred, np.ndarray):
        ones = np.ones_like(pred)
    else:
        ones = torch.ones_like(pred)

    one_minus_tgt = ones - tgt

    TP = (pred*tgt).sum()  # Postiv
    FP = (pred*one_minus_tgt).sum()  # Negatives that are in prediction

    return TP/(TP + FP)


def recall(pred, tgt):
    return sensitivity(pred, tgt)


def sensitivity(pred, tgt):
    '''
    True positive rate, how many positives are actually positive
    Supports torch or numpy
    '''
    return (pred*tgt).sum() / tgt.sum()


def specificity(pred, tgt):
    '''
    True negative rate, how many negatives are actually negative
    Doesnt work well with too many true negatives
    '''
    assert pred.shape == tgt.shape
    if isinstance(pred, np.ndarray):
        ones = np.ones_like(pred)
    else:
        ones = torch.ones_like(pred)

    ones_minus_tgt = ones - tgt
    ones_minus_pred = ones - pred

    return ((ones_minus_pred)*(ones_minus_tgt)).sum() / ones_minus_tgt.sum()


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    Haussdorf distance, from github -> LIVIAETS/surface-loss
    '''
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res


def vol_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of volume
    '''
    # q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum()
    tflat_sum = tflat.sum()

    if iflat_sum.item() == 0.0 and tflat_sum.item() == 0.0:
        print("DICE Metric got black mask and prediction!")
        dice = torch.tensor(1.0, requires_grad=True, device=inpt.device)
    else:
        dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


def batch_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of a batch of two binary masks
    Returns mean dice of all slices
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(q, -1)
    tflat = target.contiguous().view(q, -1)
    intersection = (iflat * tflat).sum(dim=1)

    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum(dim=1)
    tflat_sum = tflat.sum(dim=1)

    dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    dice = dice.mean()
    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


class DICEMetric():
    '''
    Calculates DICE Metric
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5, skip_ths=False, per_channel_metric=False, one_hot_one_class=False):
        self.apply_sigmoid = apply_sigmoid
        self.mask_ths = mask_ths
        self.skip_ths = skip_ths
        self.per_channel_metric = per_channel_metric
        self.one_hot_one_class = one_hot_one_class
        print(("DICE Metric initialized with apply_sigmoid={}, mask_ths={}, skip_ths={}, "
               "per_channel_metric{}, one_hot_one_class: {}".format(apply_sigmoid, mask_ths, skip_ths, per_channel_metric,
                                                                    one_hot_one_class)))

    def __call__(self, probs, target):
        '''
        Returns only DICE metric, as volumetric dice
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: float binary target mask
        '''
        if self.one_hot_one_class:
            probs = probs[:, 1, :, :].type(torch.float32)
            target = target[:, 1, :, :].type(torch.float32)
        else:
            probs = probs.type(torch.float32)
            target = target.type(torch.float32)

        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        # p_max = probs.max()  # removed, checked on loss, cant do this cause of weigth approach
        assert p_min >= 0.0, "FATAL ERROR: DICE metric input not positive! Did you apply sigmoid?"

        # mask = ((probs - p_min)/(p_max - p_min) > self.mask_ths).float() # were giving bad results
        if self.skip_ths:
            mask = probs
        else:
            mask = (probs > self.mask_ths).float()

        if self.per_channel_metric:
            assert len(target.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                            "volumes")
            nchannels = target.shape[1]
            return [vol_dice(mask[:, c], target[:, c], smooth=0.0).item() for c in range(nchannels)]
        else:
            return vol_dice(mask, target, smooth=0.0).item()


class DICELoss(nn.Module):
    '''
    Calculates DICE Loss
    '''
    def __init__(self, size_average=True, apply_sigmoid=False, volumetric=False, negative_loss=False, multi_target=False):
        super(DICELoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.volumetric = volumetric
        self.cross_entropy = False
        self.negative_loss = negative_loss
        self.multi_target = multi_target
        if multi_target:
            assert negative_loss
        print("DICE Loss initialized with multi_target={}, "
              "apply_sigmoid={}, volumetric={}, negative? {}".format(multi_target, apply_sigmoid, self.volumetric, negative_loss))

    def forward(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        p_max = probs.max()
        assert p_max <= 1.0 and p_min >= 0.0, "FATAL ERROR: DICE loss input not bounded! Did you apply sigmoid?"

        score = 0
        if self.volumetric:
            if self.multi_target:
                for channel in range(targets.shape[1]):
                    score += vol_dice(probs[:, channel], targets[:, channel])
            else:
                score = vol_dice(probs, targets)
        else:
            if self.multi_target:
                for channel in range(targets.shape[1]):
                    score += batch_dice(probs[:, channel], targets[:, channel])
            else:
                score = batch_dice(probs, targets)

        if self.negative_loss:
            loss = -score
        else:
            loss = 1 - score

        return loss


class JointLoss(nn.Module):
    '''
    Learns segmentation and classification jointly!
    '''
    def __init__(self, dice_weight=1, ce_weight=1, size_average=True, apply_sigmoid=False, volumetric=False, negative_loss=False):
        super(JointLoss, self).__init__()
        self.dice = DICELoss(size_average=size_average, apply_sigmoid=apply_sigmoid, volumetric=volumetric,
                             negative_loss=negative_loss)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cross_entropy = False

    def forward(self, data, target):
        if len(data[1].shape) == 1:
            data = (data[0], data[1].unsqueeze(0))
        return self.dice_weight*self.dice(data[0], target[0]) + self.ce_weight*self.ce(data[1], target[1])


class JointMetric():
    '''
    Calculates DICE and classification metrics
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5, skip_ths=False, per_channel_metric=False):
        self.dice = DICEMetric(apply_sigmoid=apply_sigmoid, mask_ths=mask_ths, skip_ths=skip_ths,
                               per_channel_metric=per_channel_metric)

    def __call__(self, data, target):
        seg_data, label_data, _ = data
        seg_target, label_target = target

        if len(label_data.shape) == 1:
            label_data = label_data.unsqueeze(0)

        dice = self.dice(seg_data, seg_target)

        acc = label_target.eq(label_data.max(dim=1)[1].long()).sum().item() / len(label_target)

        return dice, acc


class GeneralizedDice():
    '''
    Code from Boundary loss for highly unbalanced segmentation
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.cross_entropy = False
        self.loss = kwargs["loss"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        if self.loss:
            divided = 1 - divided

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss


class BoundaryLoss(nn.Module):
    def __init__(self, **kwargs):
        '''
        Init weights refers to weights for: GDL, Surface
        Ideally, increment should be called once per epoch
        '''
        super(BoundaryLoss, self).__init__()
        self.idc: List[int] = kwargs["idc"]
        self.max_ncalls: int = kwargs["max_ncalls"]
        self.use_gdl = kwargs["use_gdl"]
        self.init_weights = np.array(kwargs["init_weights"])
        self.weights = self.init_weights

        assert self.weights.sum() == 1.0
        if self.use_gdl:
            self.dice_loss = GeneralizedDice(idc=kwargs["idc"], loss=True)
            print("Using GDL for Boundary Loss")
        else:
            self.dice_loss = DICELoss()
            print("Using DICELoss for Boundary Loss")

        self.surface = SurfaceLoss(idc=kwargs["idc"])

        increment_module = abs(self.weights[0] - self.weights[1])/self.max_ncalls
        argmax = self.weights.argmax()

        self.increment = np.array([increment_module, increment_module])
        self.increment[argmax] = -1*self.increment[argmax]

        print(f"Initialized {self.__class__.__name__} with {kwargs}. Per epoch increment: {self.increment}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, target: Tensor) -> Tensor:
        dl_weight, surface_weight = self.weights
        if self.use_gdl:
            dice_probs = probs
            dice_target = target
        else:
            # Get only structure activations for normal DICELoss
            dice_probs = probs[:, 1, :, :].float()
            dice_target = target[:, 1, :, :].float()

        return dl_weight*self.dice_loss(dice_probs, dice_target) + surface_weight*self.surface(probs, dist_maps, target)

    def increment_weights(self):
        pre_increment = self.weights
        self.weights = self.weights + self.increment
        print("Boundary Loss weights: {} -> {}".format(pre_increment, self.weights))

    def reset_weights(self):
        print("Boundary loss weights reset.")
        self.weights = self.init_weights


class CELoss(nn.Module):
    '''
    Overload of torches CrossEntropyLoss, mutating one hot target to classification target
    '''
    def __init__(self, nlabels, device):
        super(CELoss, self).__init__()
        # Giving low weight to background
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1 if i == 0 else 1 for i in range(nlabels)], device=device))
        self.cross_entropy = True

    def forward(self, probs, targets):
        return self.loss(probs, targets.max(dim=1)[1].long())


def dice_test(batch_size=1, waitms=0, show=True, gdl=False):
    '''
    Randomly create masks and show dice, looking for inconsistencys
    '''
    print("Performing random rectangles test")
    print("Press ESC to interrupt test")
    key = None
    size = (400, 400)  # width, height format
    if gdl:
        print("Using generalized DICE! Needs logit output")
        dice_loss = GeneralizedDice(idc=[1, 1], loss=True)
        dice_metric = GeneralizedDice(idc=[1, 1], loss=False)
    else:
        dice_loss = DICELoss()
        dice_metric = DICEMetric()

    totensor = ToTensor(transform_to_onehot=gdl, C=2)
    while key != ESC:
        try:
            # batch_size, num_channels nrows, ncols
            batch_a = torch.zeros((batch_size, 2, size[1], size[0]))
            batch_b = torch.zeros((batch_size, 2, size[1], size[0]))
            for i in range(batch_size):
                # Make class representation
                a = np.zeros((size[1], size[0]))
                b = np.zeros((size[1], size[0]))

                x, y, w, h = random_rectangle(size)
                rect_a = (x, y, w, h)
                a[y:y+h, x:x+w] = 1

                x, y, w, h = random_rectangle(size)
                rect_b = (x, y, w, h)
                b[y:y+h, x:x+w] = 1

                if show:
                    cv.imshow("A mask", a)
                    cv.imshow("B mask", b)

                _, torch_a = totensor(a, a)
                _, torch_b = totensor(b, b)
                batch_a[i] = torch_a
                batch_b[i] = torch_b

            # Convert to onehot if needed
            print("Shape: {} / {}".format(batch_a.shape, batch_b.shape))
            metric = dice_metric(batch_a, batch_b)
            loss = dice_loss(batch_a, batch_b)

            print("Batch size: {}, loss: {}, metric: {}".format(batch_size, loss, metric))

            if show:
                display = np.zeros((size))

                x, y, w, h = rect_a
                cv.rectangle(display, (x, y), (x+w, y+h), 0.5)

                x, y, w, h = rect_b
                cv.rectangle(display, (x, y), (x+w, y+h), 1)

                text_row = 9*size[1]//10
                cv.putText(display, "m:" + str(round(metric.item(), 2)), (size[0]//10, text_row), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv.putText(display, "l:" + str(round(loss.item(), 2)), (5*size[0]//10, text_row), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv.imshow("DICE Test, m is metric and l is smooth loss", display)
                key = cv.waitKey(waitms)
        except KeyboardInterrupt:
            quit()

    print("DICE test interrupted by ESC")


def random_test():
    print("Performing random DICE test looking for inconsistencies... Press CTRL+C to continue to next test.")
    while True:
        try:
            a = torch.rand((30, 1, 128, 128))
            b = torch.rand((30, 1, 128, 128))

            dice = batch_dice(a, b)
            print(dice)
            assert dice >= 0 and dice <= 1.0
        except KeyboardInterrupt:
            return


def vol_test():
    '''
    Specific dice test for the volumetric case
    '''
    dice = DICEMetric()
    db = NatHIP(group='all', orientation=None, mode="all", verbose=True,
                transform=Compose([CenterCrop(160, 160, 160), ToTensor()]), fold=None, e2d=False, return_onehot=False)
    _, mask = random.choice(db)

    test_shape = torch.randn_like(mask)
    test_shape = (test_shape - test_shape.min()) / (test_shape.max() - test_shape.min())

    # viewnii(mask.squeeze().numpy(), test_shape.squeeze().numpy(), border_only=False)

    print("random dice {}".format(dice(test_shape, mask)))
    print("perfect dice {}".format(dice(mask, mask)))
    print("one pred {}".format(dice(torch.ones_like(mask), mask)))
    print("zero pred {}".format(dice(torch.zeros_like(mask), mask)))

    print("random sensitivity {}".format(sensitivity(test_shape, mask)))
    print("perfect sensitivity {}".format(sensitivity(mask, mask)))
    print("one pred sensitivity{}".format(sensitivity(torch.ones_like(mask), mask)))
    print("zero pred sensitivity{}".format(sensitivity(torch.zeros_like(mask), mask)))

    print("random specificity {}".format(specificity(test_shape, mask)))
    print("perfect specificity {}".format(specificity(mask, mask)))
    print("one pred specificity{}".format(specificity(torch.ones_like(mask), mask)))
    print("zero pred specificity{}".format(specificity(torch.zeros_like(mask), mask)))


if __name__ == "__main__":
    if len(argv) > 1:
        if argv[1] == "random":
            random_test()
        elif argv[1] == "rect":
            dice_test(batch_size=10, show=False)
        elif argv[1] == "rect_gdl":
            dice_test(batch_size=1, show=True, gdl=True)
        elif argv[1] == "volumetric":
            vol_test()
        else:
            print("Unrecognized arguments {}".format(str(argv)))
    else:
        random_test()
        dice_test()
