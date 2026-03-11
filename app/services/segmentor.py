import mediapipe as mp

_segmentor = None


def get_segmentor():
    global _segmentor
    if _segmentor is None:
        _segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return _segmentor