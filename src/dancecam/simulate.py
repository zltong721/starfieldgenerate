"""Explicitly approximate (not original) DanceCam short-exposure simulator."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


ARCSEC_TO_RAD = np.pi / (180 * 3600)


def sample_stars(cfg, rng):
    n = int(rng.integers(cfg["star_count_min"], cfg["star_count_max"] + 1))
    hi, lo = cfg["magnitude_max"], cfg["magnitude_min"]
    # Inverse CDF of p(m) proportional to 10^(alpha*(m-hi)).
    alpha = cfg["magnitude_alpha"] * np.log(10)
    u = rng.random(n)
    mags = hi + np.log(np.exp(alpha * (lo - hi)) + u * (1 - np.exp(alpha * (lo - hi)))) / alpha
    xy = rng.uniform(0, cfg["image_size"] - 1, (n, 2))
    flux = cfg["electrons_at_mag15"] * 10 ** (-0.4 * (mags - 15))
    return np.column_stack((xy, mags, flux)).astype(np.float32)


def phase_screen(n, pitch_m, r0_m, outer_m, inner_m, rng):
    """Modified von Karman phase, with approximate PSD normalization."""
    return phase_screen_rect(n, n, pitch_m, r0_m, outer_m, inner_m, rng)


def phase_screen_rect(nx, ny, pitch_m, r0_m, outer_m, inner_m, rng):
    """Rectangular screen allows long non-wrapping frozen-flow trajectories."""
    fx = fftfreq(nx, d=pitch_m).astype(np.float32)[None, :]
    fy = fftfreq(ny, d=pitch_m).astype(np.float32)[:, None]
    f2 = fx * fx + fy * fy
    psd = 0.023 * r0_m ** (-5 / 3) * np.exp(-f2 * inner_m ** 2) * (f2 + outer_m ** -2) ** (-11 / 6)
    psd[0, 0] = 0
    white = rng.normal(size=(ny, nx)).astype(np.float32)
    phase = ifft2(fft2(white) * np.sqrt(psd) / pitch_m).real
    return phase.astype(np.float32)


def _sample_screen(screen, x_m, y_m, shift_x_m, shift_y_m, pitch_m):
    ny, nx = screen.shape
    coords = np.array(((y_m + shift_y_m) / pitch_m + ny / 2,
                       (x_m + shift_x_m) / pitch_m + nx / 2))
    return map_coordinates(screen, coords, order=1, mode="wrap")


def _pupil(cfg):
    n = cfg.get("propagation_grid", cfg["pupil_grid"])
    pitch = cfg["aperture_m"] / cfg["pupil_grid"]
    axis = (np.arange(n) - (n - 1) / 2) * pitch
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)
    aperture = ((radius <= cfg["aperture_m"] / 2) &
                (radius >= cfg["aperture_m"] * cfg["obstruction_ratio"] / 2)).astype(np.float32)
    return aperture, xx, yy, pitch


def _psf(cfg, screens, winds, t, angle_xy, ideal=False):
    pupil, xx, yy, pitch = _pupil(cfg)
    wave_m = cfg["wavelength_nm"] * 1e-9
    radius = np.hypot(xx, yy)
    inner, outer = 0.75 * cfg["aperture_m"], cfg["aperture_m"]
    window = np.where(radius <= inner, 1,
                      np.where(radius >= outer, 0,
                               0.5 * (1 + np.cos(np.pi * (radius - inner) / (outer - inner)))))
    field = (pupil if ideal else window).astype(np.complex64)
    if not ideal:
        # High layers first. Angular shift of a line of sight is height*angle.
        for idx, (height, screen, wind) in enumerate(zip(cfg["layer_heights_m"], screens, winds)):
            sx = cfg.get("screen_start_x_m", 0.0) + height * angle_xy[0] + t * wind[0]
            sy = height * angle_xy[1] + t * wind[1]
            phase = _sample_screen(screen, xx, yy, sx, sy, pitch)
            field *= np.exp(1j * phase)
            if idx + 1 < len(screens) and height > cfg["layer_heights_m"][idx + 1]:
                dz = height - cfg["layer_heights_m"][idx + 1]
                ff = fftfreq(field.shape[0], pitch)
                f2 = ff[:, None] ** 2 + ff[None, :] ** 2
                field = ifft2(fft2(field) * np.exp(-1j * np.pi * wave_m * dz * f2))
        field *= pupil
    intensity = np.fft.fftshift(np.abs(fft2(field)) ** 2)
    # Convert diffraction-plane sample spacing to detector pixel spacing.
    stamp = cfg["psf_stamp"]
    desired = np.arange(stamp) - (stamp - 1) / 2
    diffraction_arcsec = wave_m / (intensity.shape[0] * pitch) / ARCSEC_TO_RAD
    step = cfg["pixel_scale_arcsec"] / diffraction_arcsec
    result = np.zeros((stamp, stamp), dtype=np.float64)
    for offset_y in (-.375, -.125, .125, .375):
        for offset_x in (-.375, -.125, .125, .375):
            yy2, xx2 = np.meshgrid((desired + offset_y) * step + intensity.shape[0] / 2,
                                   (desired + offset_x) * step + intensity.shape[1] / 2,
                                   indexing="ij")
            result += map_coordinates(intensity, [yy2, xx2], order=1, mode="constant", cval=0)
    result /= 16
    result = np.maximum(result, 0)
    return (result / max(result.sum(), 1e-20)).astype(np.float32)


def _deposit(stars, size):
    image = np.zeros((size, size), dtype=np.float32)
    for x, y, _mag, flux in stars:
        ix, iy = int(x), int(y)
        dx, dy = x - ix, y - iy
        for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                               (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
            if ix + ox < size and iy + oy < size:
                image[iy + oy, ix + ox] += flux * weight
    return image


def simulate_one(cfg, rng, stars=None):
    if stars is None:
        stars = sample_stars(cfg, rng)
    stars = np.asarray(stars, dtype=np.float32)
    size = cfg["image_size"]
    scene = _deposit(stars, size)
    pupil_pitch = cfg["aperture_m"] / cfg["pupil_grid"]
    screens, winds = [], []
    for mean_r0, sigma_r0, mean_w, sigma_w in zip(
        cfg["layer_r0_mean_m"], cfg["layer_r0_sigma_m"],
        cfg["layer_wind_mean_mps"], cfg["layer_wind_sigma_mps"]
    ):
        r0 = max(0.03, rng.normal(mean_r0, sigma_r0))
        screens.append(phase_screen(cfg["screen_grid"], pupil_pitch, r0,
                                    cfg["outer_scale_m"], cfg["inner_scale_m"], rng))
        speed = max(0, rng.normal(mean_w, sigma_w))
        theta = rng.uniform(0, 2*np.pi)
        winds.append((speed * np.cos(theta), speed * np.sin(theta)))
    target = fftconvolve(scene, _psf(cfg, screens, winds, 0, (0, 0), ideal=True), mode="same")
    frames = np.empty((cfg["n_frames"], size, size), dtype=np.float32)
    # 2x2 isoplanatic zones are a tractable wide-field proxy; original field model is unavailable.
    zone_count = 2 if size >= 64 else 1
    deposits = []
    for zy in range(zone_count):
        for zx in range(zone_count):
            sub = stars[(stars[:, 0] >= zx * size / zone_count) &
                        (stars[:, 0] < (zx + 1) * size / zone_count) &
                        (stars[:, 1] >= zy * size / zone_count) &
                        (stars[:, 1] < (zy + 1) * size / zone_count)]
            angle = ((zx + .5 - zone_count / 2) * size / zone_count * cfg["pixel_scale_arcsec"] * ARCSEC_TO_RAD,
                     (zy + .5 - zone_count / 2) * size / zone_count * cfg["pixel_scale_arcsec"] * ARCSEC_TO_RAD)
            deposits.append((_deposit(sub, size), angle))
    for i in range(cfg["n_frames"]):
        t = i / cfg["frame_rate_hz"]
        image = np.zeros((size, size), dtype=np.float32)
        for zone, angle in deposits:
            if zone.any():
                image += fftconvolve(zone, _psf(cfg, screens, winds, t, angle), mode="same")
        noisy = rng.poisson(np.maximum(image + cfg["sky_e_per_pixel"], 0)).astype(np.float32)
        noisy += rng.normal(0, cfg["read_noise_e"], noisy.shape).astype(np.float32)
        frames[i] = noisy
    return frames, np.maximum(target, 0).astype(np.float32), stars


def save_sequence_fits(path, video, target, catalog, cfg):
    """Export one sequence with the layout expected by dancecam.infer."""
    from astropy.io import fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU(np.asarray(video, dtype=np.float32))
    for key, value in {
        "BUNIT": "electron", "PIXSCALE": cfg["pixel_scale_arcsec"],
        "WAVE_NM": cfg["wavelength_nm"], "DIAM_M": cfg["aperture_m"],
        "CADENCE": 1 / cfg["frame_rate_hz"], "SIMSEED": cfg["seed"],
    }.items():
        primary.header[key] = value
    primary.header.add_history("Approximate DanceCam simulation; not original paper data.")
    truth = fits.ImageHDU(np.asarray(target, dtype=np.float32), name="GROUNDTRUTH")
    truth.header["BUNIT"] = "electron"
    sources = fits.BinTableHDU.from_columns([
        fits.Column(name=name, format="E", array=np.asarray(catalog)[:, i])
        for i, name in enumerate(("X", "Y", "MAG", "FLUX"))
    ], name="SOURCES")
    sources.header.add_history("X and Y use zero-based pixel coordinates.")
    config_json = json.dumps(cfg, ensure_ascii=True)
    config = fits.BinTableHDU.from_columns([
        fits.Column(name="JSON", format=f"{len(config_json)}A", array=[config_json])
    ], name="CONFIG")
    fits.HDUList([primary, truth, sources, config]).writeto(path, overwrite=True, checksum=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fits-dir", type=Path,
                        help="Also export each sequence as FITS under this directory; retain NPZ output.")
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--allow-paper-scale", action="store_true")
    parser.add_argument("--stream", action="store_true",
                        help="Write one NPZ per sequence to an output directory, avoiding RAM accumulation.")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    count = args.limit_sequences or cfg["n_sequences"]
    if count > 100 and not args.allow_paper_scale:
        parser.error("Over 100 sequences is expensive; pass --allow-paper-scale explicitly.")
    if count > 100 and not args.stream:
        parser.error("Over 100 sequences also requires --stream to avoid unbounded RAM use.")
    if args.stream and args.out.suffix:
        parser.error("With --stream, --out must be a directory (without a file extension).")
    if cfg["pupil_grid"] > cfg["screen_grid"]:
        parser.error("screen_grid must be >= pupil_grid")
    rng = np.random.default_rng(cfg["seed"])
    videos, targets, catalogs = [], [], []
    if args.stream:
        if args.out.exists() and any(args.out.glob("sequence_*.npz")):
            parser.error(f"Output directory already contains sequences: {args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        video, target, catalog = simulate_one(cfg, rng)
        if args.fits_dir is not None:
            save_sequence_fits(args.fits_dir / f"sequence_{idx:06d}.fits", video, target, catalog, cfg)
        if args.stream:
            np.savez_compressed(args.out / f"sequence_{idx:06d}.npz", video=video,
                                target=target, catalog=catalog,
                                config_json=json.dumps(cfg, ensure_ascii=False))
        else:
            videos.append(video)
            targets.append(target)
            catalogs.append(catalog)
        print(f"sequence {idx+1}/{count}: {len(catalog)} stars", flush=True)
    if not args.stream:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, video=np.stack(videos), target=np.stack(targets),
                            catalogs=np.array(catalogs, dtype=object),
                            config_json=json.dumps(cfg, ensure_ascii=False))
    print(args.out)


if __name__ == "__main__":
    main()
