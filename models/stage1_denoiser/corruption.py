"""
corruption.py
=============
Synthetic optical and civil infrastructure sensor corruption pipeline.

Supports:
1. Legacy SAR speckle noise (Rayleigh and Gamma multiplicative distributions).
2. Civil infrastructure inspection artifacts per PRD:
   - Illumination gradients (tunnel vignetting / directional LED falloff)
   - Water droplets (localized refractive blur + specular rim highlights)
   - Vibration motion blur (directional linear PSF simulating drone/crawler vibrations)
   - Specular glare (saturated highlights on wet concrete / metallic pipe surfaces)
"""

import cv2
import numpy as np


def apply_rayleigh_speckle(
    image: np.ndarray, scale: float = 0.3, rng: np.random.Generator = None
) -> np.ndarray:
    """Apply multiplicative Rayleigh-distributed speckle noise."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.rayleigh(scale=scale, size=image.shape).astype(np.float32)
    noise = noise / (scale * np.sqrt(np.pi / 2))
    corrupted = image * noise
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def apply_gamma_speckle(
    image: np.ndarray, num_looks: float = 3.0, rng: np.random.Generator = None
) -> np.ndarray:
    """Apply multiplicative Gamma-distributed speckle noise."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.gamma(
        shape=num_looks, scale=1.0 / num_looks, size=image.shape
    ).astype(np.float32)
    corrupted = image * noise
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def corrupt_image(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Legacy speckle corruption."""
    if rng is None:
        rng = np.random.default_rng()

    mode = rng.choice(["rayleigh", "gamma", "both"])
    if mode == "rayleigh":
        scale = rng.uniform(0.15, 0.5)
        return apply_rayleigh_speckle(image, scale=scale, rng=rng)
    elif mode == "gamma":
        num_looks = rng.uniform(1.0, 5.0)
        return apply_gamma_speckle(image, num_looks=num_looks, rng=rng)
    else:
        scale = rng.uniform(0.15, 0.35)
        num_looks = rng.uniform(2.0, 5.0)
        result = apply_rayleigh_speckle(image, scale=scale, rng=rng)
        result = apply_gamma_speckle(result, num_looks=num_looks, rng=rng)
        return result


# ---------------------------------------------------------------------------
# Civil Infrastructure Optical Corruptions (PRD Section 2)
# ---------------------------------------------------------------------------

def apply_illumination_gradient(
    image: np.ndarray, rng: np.random.Generator = None
) -> np.ndarray:
    """Simulates uneven lighting, tunnel vignetting, or directional LED falloff."""
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    # Random gradient direction and center
    center_x = rng.uniform(0.2, 0.8) * w
    center_y = rng.uniform(0.2, 0.8) * h
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Radial or linear gradient
    if rng.random() > 0.5:
        # Radial falloff
        max_dist = np.hypot(max(center_x, w - center_x), max(center_y, h - center_y))
        dist = np.hypot(x_coords - center_x, y_coords - center_y) / (max_dist + 1e-6)
        min_factor = rng.uniform(0.25, 0.55)
        gradient = 1.0 - (1.0 - min_factor) * dist
    else:
        # Directional linear gradient
        angle = rng.uniform(0, 2 * np.pi)
        proj = np.cos(angle) * (x_coords - w / 2) + np.sin(angle) * (y_coords - h / 2)
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
        min_factor = rng.uniform(0.3, 0.6)
        gradient = min_factor + (1.0 - min_factor) * proj

    gradient = gradient.astype(np.float32)
    if image.ndim == 3:
        gradient = gradient[:, :, np.newaxis]

    corrupted = image * gradient
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def apply_water_droplets(
    image: np.ndarray, rng: np.random.Generator = None, num_droplets: int = None
) -> np.ndarray:
    """Simulates localized condensation, water droplets, and refractive rim distortion."""
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    if num_droplets is None:
        num_droplets = rng.integers(3, 10)

    result = image.copy()
    for _ in range(num_droplets):
        cx = int(rng.integers(int(w * 0.1), int(w * 0.9)))
        cy = int(rng.integers(int(h * 0.1), int(h * 0.9)))
        radius = int(rng.integers(max(4, int(min(h, w) * 0.03)), max(8, int(min(h, w) * 0.10))))

        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)

        # Region mask for droplet
        y_grid, x_grid = np.ogrid[y1:y2, x1:x2]
        dist = np.hypot(x_grid - cx, y_grid - cy)
        mask = dist <= radius

        if not np.any(mask):
            continue

        # Droplet interior: mild blur and refraction (magnification / displacement)
        patch = result[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(patch, (7, 7), 1.8)

        # Droplet rim highlight (specular reflection on meniscus boundary)
        rim_mask = (dist > radius * 0.75) & (dist <= radius)
        highlight = 0.35 * (1.0 - (radius - dist) / (radius * 0.25 + 1e-6))
        highlight = np.clip(highlight, 0.0, 0.5).astype(np.float32)

        if patch.ndim == 3:
            mask_3d = mask[:, :, np.newaxis]
            rim_3d = rim_mask[:, :, np.newaxis]
            hl_3d = highlight[:, :, np.newaxis]
            result[y1:y2, x1:x2] = np.where(mask_3d, blurred * 0.85 + hl_3d * rim_3d, patch)
        else:
            result[y1:y2, x1:x2] = np.where(mask, blurred * 0.85 + highlight * rim_mask, patch)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_vibration_motion_blur(
    image: np.ndarray, rng: np.random.Generator = None, kernel_size: int = None
) -> np.ndarray:
    """Simulates mechanical vibration motion blur along random linear trajectory."""
    if rng is None:
        rng = np.random.default_rng()

    if kernel_size is None:
        kernel_size = int(rng.choice([5, 7, 9, 11]))
    if kernel_size % 2 == 0:
        kernel_size += 1

    angle_deg = rng.uniform(0, 180)
    # Generate 1D motion kernel and rotate
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0

    rot_mat = cv2.getRotationMatrix2D(
        (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle_deg, 1.0
    )
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        kernel[kernel_size // 2, kernel_size // 2] = 1.0

    blurred = cv2.filter2D(image, -1, kernel)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def apply_specular_glare(
    image: np.ndarray, rng: np.random.Generator = None
) -> np.ndarray:
    """Simulates localized high-intensity specular reflection on wet or metallic surface."""
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    glare_x = rng.uniform(0.15, 0.85) * w
    glare_y = rng.uniform(0.15, 0.85) * h
    sigma_x = rng.uniform(0.08, 0.22) * w
    sigma_y = rng.uniform(0.08, 0.22) * h
    peak_intensity = rng.uniform(0.5, 0.95)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    glare = peak_intensity * np.exp(
        -(((x_coords - glare_x) ** 2) / (2 * sigma_x ** 2) + ((y_coords - glare_y) ** 2) / (2 * sigma_y ** 2))
    )
    glare = glare.astype(np.float32)
    if image.ndim == 3:
        glare = glare[:, :, np.newaxis]

    corrupted = image + glare
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def corrupt_civil_image(
    image: np.ndarray, rng: np.random.Generator = None
) -> np.ndarray:
    """Chains randomized civil infrastructure optical corruptions."""
    if rng is None:
        rng = np.random.default_rng()

    corruptions = [
        apply_illumination_gradient,
        apply_water_droplets,
        apply_vibration_motion_blur,
        apply_specular_glare,
    ]
    # Pick 1 to 3 corruptions to apply
    k = int(rng.integers(1, 4))
    chosen = rng.choice(corruptions, size=k, replace=False)

    corrupted = image.copy()
    for fn in chosen:
        corrupted = fn(corrupted, rng=rng)
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def load_and_normalize(
    path: str, grayscale: bool = False, target_size: tuple = None
) -> np.ndarray:
    """Loads an image from disk, converts to RGB (or grayscale), and normalizes to [0, 1] float32."""
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        normalized = img.astype(np.float32) / 255.0
    else:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        # Convert BGR from disk to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        normalized = img_rgb.astype(np.float32) / 255.0

    if target_size is not None:
        w, h = target_size
        # Sizing guard: enforce divisible by 4
        w = (w // 4) * 4
        h = (h // 4) * 4
        normalized = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_AREA)

    return normalized


def generate_pair(
    clean: np.ndarray, rng: np.random.Generator = None, civil: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates (corrupted, clean) training pair."""
    if civil:
        corrupted = corrupt_civil_image(clean, rng=rng)
    else:
        corrupted = corrupt_image(clean, rng=rng)
    return corrupted, clean
