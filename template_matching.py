"""OpenCV matching primitives; no state updates or file output."""
import cv2
import numpy as np


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError('image must be a non-empty NumPy array')
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('image must be uint8 grayscale or BGR')
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError('color image must have three BGR channels')


def crop_at(image, origin, size):
    """Clip a rectangle before slicing, including fully offscreen rectangles."""
    h, w = image.shape[:2]
    x1, y1 = np.floor(origin).astype(int)
    x2, y2 = np.ceil(np.asarray(origin) + size).astype(int)
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    return image[y1:y2, x1:x2], np.array([x1, y1], dtype=float)


def _match_template(search, template):
    if search.size == 0 or template.size == 0:
        return -1., (0, 0)
    if any(a < b for a, b in zip(search.shape[:2], template.shape[:2])):
        return -1., (0, 0)
    # CCOEFF_NORMED reports 1 for a spatially constant template. Reject it.
    if float(np.sum(np.var(template.astype(float), axis=(0, 1)))) < 1e-8:
        return -1., (0, 0)
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
    _, score, _, location = cv2.minMaxLoc(response)
    return float(np.clip(score, -1, 1)), location


def multi_scale_match(search_region, template, base_w, base_h, scales):
    best = (-1., 0, 0, 1.)
    h, w = search_region.shape[:2]
    for scale in scales:
        tw, th = max(3, int(base_w * scale)), max(3, int(base_h * scale))
        if tw > w or th > h:
            continue
        score, (x, y) = _match_template(search_region, cv2.resize(template, (tw, th)))
        if score > best[0]:
            best = score, x, y, float(scale)
    return best


def dense_scale_match(search_region, template, base_w, base_h, scales=None):
    scales = np.linspace(.5, 1.5, 50) if scales is None else np.asarray(scales, dtype=float)
    if len(scales) == 0:
        raise ValueError('scales cannot be empty')
    scores, locations = [], []
    for scale in scales:
        score, x, y, _ = multi_scale_match(search_region, template, base_w, base_h, [scale])
        scores.append(score)
        locations.append((x, y))
    index = int(np.argmax(scores))
    scale = float(scales[index])
    if 0 < index < len(scales) - 1:
        left, peak, right = scores[index-1:index+2]
        denominator = 2 * (2 * peak - left - right)
        if peak > left and peak > right and abs(denominator) > 1e-8:
            scale += (right - left) / denominator * (scales[index+1] - scales[index])
    return scores[index], *locations[index], float(np.clip(scale, scales.min(), scales.max()))


def full_image_search(img, template, base_w, base_h, search_scale):
    if not np.isfinite(search_scale) or not 0 < search_scale <= 1:
        raise ValueError('search_scale must be in (0, 1]')
    h, w = img.shape[:2]
    sw, sh = max(1, round(w * search_scale)), max(1, round(h * search_scale))
    small = cv2.resize(img, (sw, sh))
    tw, th = max(3, round(base_w * sw/w)), max(3, round(base_h * sh/h))
    score, (x, y) = _match_template(small, cv2.resize(template, (tw, th)))
    return score, int(x * w/sw), int(y * h/sh), 1.
