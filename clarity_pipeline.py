import os
import urllib.request
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clarity")
REAL_ESRGAN_X2_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth"
)
REAL_ESRGAN_X2_WEIGHTS = os.path.join(MODEL_DIR, "RealESRGAN_x2plus.pth")

_REALESRGAN_UPSAMPLER = None
_REALESRGAN_INIT_ERROR = None
_SWIN2SR_MODEL = None
_SWIN2SR_PROCESSOR = None
_SWIN2SR_INIT_ERROR = None


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _download_weights_if_missing() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(REAL_ESRGAN_X2_WEIGHTS):
        return
    urllib.request.urlretrieve(REAL_ESRGAN_X2_URL, REAL_ESRGAN_X2_WEIGHTS)


def _get_realesrgan_upsampler():
    global _REALESRGAN_UPSAMPLER, _REALESRGAN_INIT_ERROR
    if _REALESRGAN_UPSAMPLER is not None:
        return _REALESRGAN_UPSAMPLER
    if _REALESRGAN_INIT_ERROR is not None:
        return None

    try:
        _download_weights_if_missing()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        _REALESRGAN_UPSAMPLER = RealESRGANer(
            scale=2,
            model_path=REAL_ESRGAN_X2_WEIGHTS,
            model=model,
            tile=200,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        return _REALESRGAN_UPSAMPLER
    except Exception as ex:
        _REALESRGAN_INIT_ERROR = str(ex)
        return None


def _classical_detail_boost(local_img: Image.Image) -> Image.Image:
    """
    Physics-informed adaptive restoration:
    1) pre-condition denoise
    2) UDCP-style dehaze (B/G channels)
    3) color balance + adaptive gamma
    4) LAB-space guarded sharpening
    5) mild multi-branch fusion for stability
    """
    bgr = _pil_to_bgr(local_img)
    h, w = bgr.shape[:2]
    is_low_res = max(h, w) < 800

    # 1) Adaptive pre-conditioning denoise.
    bgr = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=3 if is_low_res else 2,
        hColor=3 if is_low_res else 2,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 2) UDCP-style dehazing (use only B/G for dark channel).
    b_chan, g_chan, r_chan = cv2.split(bgr)
    dark_bg = cv2.min(b_chan, g_chan)
    dark_bg = cv2.erode(dark_bg, np.ones((15, 15), np.uint8))
    haze_degree = float(np.mean(dark_bg))
    omega = 0.95 if haze_degree > 100 else 0.75

    # Estimate atmospheric light from top dark-channel pixels.
    flat_dark = dark_bg.reshape(-1)
    flat_img = bgr.reshape(-1, 3)
    top_k = max(1, int(0.001 * flat_dark.size))
    idx = np.argpartition(flat_dark, -top_k)[-top_k:]
    a_est = np.mean(flat_img[idx], axis=0).astype(np.float32)
    a_est = np.clip(a_est, 30.0, 240.0)

    transmission = 1.0 - omega * (dark_bg.astype(np.float32) / (np.max(a_est[:2]) + 1e-6))
    transmission = cv2.GaussianBlur(transmission, (5, 5), 0)
    transmission = np.clip(transmission, 0.1, 1.0)
    t3 = transmission[..., None]
    dehazed = (bgr.astype(np.float32) - a_est) / t3 + a_est
    bgr = np.clip(dehazed, 0, 255).astype(np.uint8)

    # 3) Successive color correction + luminance enhancement.
    # Gray-world white balance.
    avg_b, avg_g, avg_r = [float(x) for x in cv2.mean(bgr)[:3]]
    avg = (avg_b + avg_g + avg_r) / 3.0
    gains = np.array([avg / (avg_b + 1e-6), avg / (avg_g + 1e-6), avg / (avg_r + 1e-6)])
    gains = np.clip(gains, 0.80, 1.30)
    bgr = np.clip(bgr.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)

    # Red compensation anchored to green channel for underwater attenuation.
    b, g, r = cv2.split(bgr.astype(np.float32))
    r = np.clip(r + 0.15 * (g - r), 0, 255)
    bgr = cv2.merge([b, g, r]).astype(np.uint8)

    # Adaptive gamma from luminance mean.
    lum_mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
    gamma = 0.8 if lum_mean < 35 else (0.9 if lum_mean < 70 else 1.0)
    if gamma != 1.0:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        bgr = cv2.LUT(bgr, lut)

    # Branch A: contrast/luminance enhancement (safe CLAHE values from doc ranges).
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_clip = 1.2 if max(h, w) < 540 else (1.5 if is_low_res else 1.8)
    tile = (4, 4) if is_low_res else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile)
    l = clahe.apply(l)
    branch_a = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4) Branch B: LAB sharpening with edge-mask guard.
    den = cv2.bilateralFilter(branch_a, d=9, sigmaColor=50, sigmaSpace=50)
    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(h * w + 1e-6)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edge_mask_3 = np.repeat((edge_mask > 0)[:, :, None], 3, axis=2)
    if edge_density > 0.01:
        den_lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l2, a2, b2 = cv2.split(den_lab)
        gauss_l = cv2.GaussianBlur(l2, (0, 0), 1.1)
        l_sharp = cv2.addWeighted(l2, 1.35, gauss_l, -0.35, 0)
        lab_sharp = cv2.cvtColor(cv2.merge([l_sharp, a2, b2]), cv2.COLOR_LAB2BGR)
        # Apply sharpening only where edges exist.
        branch_b = np.where(edge_mask_3, lab_sharp, den)
    else:
        branch_b = den

    # 5) Weighted fusion to avoid linear over-processing artifacts.
    def _contrast_map(img_bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.abs(cv2.Laplacian(g, cv2.CV_32F))

    def _saturation_map(img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.astype(np.float32)
        return np.std(img, axis=2)

    c1 = _contrast_map(branch_a) + _saturation_map(branch_a)
    c2 = _contrast_map(branch_b) + _saturation_map(branch_b)
    eps = 1e-6
    wsum = c1 + c2 + eps
    w1 = (c1 / wsum)[..., None]
    w2 = (c2 / wsum)[..., None]
    fused = np.clip(
        branch_a.astype(np.float32) * w1 + branch_b.astype(np.float32) * w2, 0, 255
    ).astype(np.uint8)

    # Controlled vibrance lift.
    hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.03, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Residual re-injection to reduce painterly smoothness.
    resid = cv2.subtract(_pil_to_bgr(local_img), cv2.GaussianBlur(_pil_to_bgr(local_img), (0, 0), 1.0))
    out = np.clip(out.astype(np.float32) + 0.05 * resid.astype(np.float32), 0, 255).astype(np.uint8)
    return _bgr_to_pil(out)


def _deep_clarity_boost(local_img: Image.Image) -> Optional[Image.Image]:
    """
    Deep clarity pass using RealESRGAN x2.
    Resizes back to original dimensions to avoid UX changes.
    """
    try:
        restored = _deep_clarity_boost_swin2sr(local_img)
        if restored is not None:
            return restored
    except Exception:
        pass

    # Fallback deep path if optional realesrgan stack exists in environment.
    upsampler = _get_realesrgan_upsampler()
    if upsampler is None:
        return None

    try:
        bgr = _pil_to_bgr(local_img)
        enhanced_bgr, _ = upsampler.enhance(bgr, outscale=1.5)
        restored = _bgr_to_pil(enhanced_bgr).resize(local_img.size, Image.LANCZOS)
        return restored
    except Exception:
        return None


def _get_swin2sr():
    global _SWIN2SR_MODEL, _SWIN2SR_PROCESSOR, _SWIN2SR_INIT_ERROR
    if _SWIN2SR_MODEL is not None and _SWIN2SR_PROCESSOR is not None:
        return _SWIN2SR_PROCESSOR, _SWIN2SR_MODEL
    if _SWIN2SR_INIT_ERROR is not None:
        return None, None

    try:
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

        model_id = "caidas/swin2SR-classical-sr-x2-64"
        _SWIN2SR_PROCESSOR = AutoImageProcessor.from_pretrained(model_id)
        _SWIN2SR_MODEL = Swin2SRForImageSuperResolution.from_pretrained(model_id)
        _SWIN2SR_MODEL.eval()
        return _SWIN2SR_PROCESSOR, _SWIN2SR_MODEL
    except Exception as ex:
        _SWIN2SR_INIT_ERROR = str(ex)
        return None, None


def _deep_clarity_boost_swin2sr(local_img: Image.Image) -> Optional[Image.Image]:
    processor, model = _get_swin2sr()
    if processor is None or model is None:
        return None

    try:
        with torch.no_grad():
            inputs = processor(images=local_img, return_tensors="pt")
            outputs = model(**inputs)
            out = outputs.reconstruction.squeeze().float().cpu().clamp_(0, 1).numpy()
        if out.ndim == 3:
            out = np.transpose(out, (1, 2, 0))
        out_img = (out * 255.0).round().astype(np.uint8)
        return Image.fromarray(out_img).resize(local_img.size, Image.LANCZOS)
    except Exception:
        return None


def build_local_clarity_candidates(local_img: Image.Image) -> Dict[str, Image.Image]:
    """
    Returns ordered candidate set; first item is always baseline local image.
    """
    candidates: Dict[str, Image.Image] = {"local": local_img}
    candidates["local_classical"] = _classical_detail_boost(local_img)
    deep = _deep_clarity_boost(local_img)
    if deep is not None:
        candidates["local_deep"] = deep
    return candidates


def build_gemini_clarity_candidates(gemini_img: Image.Image) -> Dict[str, Image.Image]:
    """
    Builds Gemini-first candidate set. Keeps raw Gemini output and adds
    restrained clarity refinements to avoid over-processing.
    """
    candidates: Dict[str, Image.Image] = {"gemini": gemini_img}
    w, h = gemini_img.size
    low_res = max(w, h) < 800

    # For low-res inputs, avoid heavy refinement that can create mushy textures.
    if not low_res:
        candidates["gemini_refined"] = _classical_detail_boost(gemini_img)
        deep = _deep_clarity_boost(gemini_img)
        if deep is not None:
            candidates["gemini_deep"] = deep
    return candidates


def _quality_score(reference: Image.Image, candidate: Image.Image) -> float:
    """
    Higher is better: combines sharpness, local contrast, and anti-artifact term.
    """
    ref = np.array(reference.convert("RGB")).astype(np.float32)
    can = np.array(candidate.convert("RGB")).astype(np.float32)

    can_gray = cv2.cvtColor(can.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(can_gray, cv2.CV_64F).var()
    contrast = can_gray.std()

    # Penalize clipping artifacts.
    clip_low = (can < 3).mean()
    clip_high = (can > 252).mean()
    clip_penalty = (clip_low + clip_high) * 200.0

    # Penalize excessive deviation from original structure (softly).
    ref_gray = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mse = np.mean((ref_gray - can_gray.astype(np.float32)) ** 2)
    structure_penalty = min(14.0, mse / 75.0)

    # Underwater quality terms (UCIQE/UIQM-inspired).
    can_u8 = can.astype(np.uint8)
    lab = cv2.cvtColor(can_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    uciqe = 0.4680 * chroma.std() + 0.2745 * (lab[:, :, 0].mean() / 100.0) + 0.2576 * (
        chroma.mean() / 128.0
    )

    r, g, b = can[:, :, 0], can[:, :, 1], can[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = -0.0268 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2) + 0.1586 * np.sqrt(
        rg.std() ** 2 + yb.std() ** 2
    )
    uism = can_gray.std() / 128.0
    uiconm = np.log(
        1.0 + np.abs(can_gray.astype(np.float32) - can_gray.mean()).mean() / (can_gray.std() + 1e-6)
    )
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm

    # Artifact guard: excessive total variation can indicate fake textures.
    tv = float(np.mean(np.abs(np.diff(can_gray.astype(np.float32), axis=0)))) + float(
        np.mean(np.abs(np.diff(can_gray.astype(np.float32), axis=1)))
    )
    tv_penalty = max(0.0, tv - 24.0) * 0.25

    return (
        (0.04 * sharpness)
        + (0.70 * contrast)
        + (6.0 * uciqe)
        + (2.8 * uiqm)
        - clip_penalty
        - structure_penalty
        - tv_penalty
    )


def choose_best_candidate(
    reference_img: Image.Image,
    candidates: Dict[str, Image.Image],
    gemini_choice: Optional[str] = None,
) -> Tuple[str, Image.Image]:
    """
    Deterministic quality gate. Gemini choice can softly bias but cannot force
    a clearly worse candidate.
    """
    best_name = None
    best_score = None

    for name, img in candidates.items():
        score = _quality_score(reference_img, img)

        if gemini_choice == "GEMINI" and "gemini" in name:
            score += 7.5
        elif gemini_choice == "LOCAL" and "local" in name:
            score += 2.5

        if best_score is None or score > best_score:
            best_name = name
            best_score = score

    return best_name, candidates[best_name]

