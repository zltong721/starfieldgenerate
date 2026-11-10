"""Resume 5,000 open-cluster Gaia views with isolated sky-area splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import numpy as np
from PIL import Image
import torch

from .cluster_video import fetch, project, read_clusters
from .fits_to_npz import convert
from .mutian_video import Renderer
from .simulate import save_sequence_fits


def split_sky_areas(clusters, downloads, seed):
    """Connect overlapping query regions and shared Gaia source IDs."""
    parent = list(range(len(clusters)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[root(i)] = root(j)

    coords = SkyCoord([c['ra'] for c in clusters] * u.deg,
                      [c['dec'] for c in clusters] * u.deg)
    for i in range(len(clusters)):
        for j in range(i):
            if coords[i].separation(coords[j]) < 6 * u.arcmin:
                union(i, j)
    owner = {}
    for i, (_, path) in enumerate(downloads):
        for source_id in Table.read(path, format='ascii.csv')['source_id']:
            source_id = int(source_id)
            if source_id in owner:
                union(i, owner[source_id])
            owner[source_id] = i
    groups = {}
    for i in range(len(clusters)):
        groups.setdefault(root(i), []).append(i)
    components = list(groups.values())
    np.random.default_rng(seed).shuffle(components)
    assignment = {}
    counts = {'train': 0, 'validation': 0, 'test': 0}
    for component in components:
        if counts['validation'] < 10:
            label = 'validation'
        elif counts['test'] < 10:
            label = 'test'
        else:
            label = 'train'
        for i in component:
            assignment[i] = label
        counts[label] += len(component)
    if not all(counts.values()):
        raise ValueError(f'Could not create all three sky-area splits: {counts}')
    return assignment, components


def preview(fits_path, png_path):
    with fits.open(fits_path, memmap=True) as h:
        frame = np.asarray(h[0].data[0], dtype=np.float32)
    low, high = np.percentile(frame, [1, 99.8])
    image = np.clip(np.arcsinh(np.maximum(frame-low, 0)) /
                    np.arcsinh(max(high-low, 1)), 0, 1)
    temporary = png_path.with_suffix('.partial.png')
    Image.fromarray(np.uint8(np.round(image*255))).save(temporary)
    temporary.replace(png_path)


def views():
    # 49 grid points plus the exact catalog centre. The grid's (0, 0) point
    # is replaced with a distinct offset, yielding 50 unique field centres.
    result = [(0., 0.)]
    result += [(float(x), float(y)) for y in np.linspace(-96, 96, 7)
               for x in np.linspace(-96, 96, 7) if x != 0 or y != 0]
    result.append((12., 20.))
    assert len(result) == 50 and len(set(result)) == 50
    return result


def write_json(path, payload):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    # On Windows, a reader briefly holding the destination can make replace
    # fail with WinError 5. Retry the atomic rename instead of aborting a run.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/mutian_193_r.json'))
    parser.add_argument('--catalog', type=Path, default=Path('data/source/catalogs/Hunt_Reffert_2024_Gaia_100_candidates.md'))
    parser.add_argument('--out', type=Path, default=Path('data/generated/opencluster'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=('train', 'validation', 'test'))
    parser.add_argument('--limit-sequences', type=int)
    args = parser.parse_args()
    if args.limit_sequences is not None and not 1 <= args.limit_sequences <= 5000:
        parser.error('--limit-sequences must be between 1 and 5000')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    clusters = read_clusters(args.catalog)
    if len(clusters) != 100:
        parser.error(f'Expected 100 catalog clusters, got {len(clusters)}')
    args.out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out).free
    missing = (args.limit_sequences or 5000) - len(list(args.out.glob('*.fits')))
    required = missing * 34 * 1024**2 + 10 * 1024**3
    print(f'Free {free/1024**3:.1f} GiB; estimated reserve {required/1024**3:.1f} GiB', flush=True)
    if free < required:
        raise OSError('Insufficient disk space for 5000 FITS/NPZ pairs')
    lock = (args.out/'opencluster_batch.lock').open('a+b')
    if os.name == 'nt':
        import msvcrt
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    partial = args.split is not None or args.limit_sequences is not None
    status_path = args.out/('batch_status_test.json' if partial else 'batch_status.json')
    cache = args.out/'gaia_cache_expanded'
    cache.mkdir(exist_ok=True)
    try:
        # The outermost requested field corner is about 170 arcsec from the
        # catalog centre. A 2.7 arcmin Gaia cone covers it with a margin.
        query_cfg = dict(cfg, image_size=1350)
        print('Fetching 100 expanded Gaia cones', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloads = list(pool.map(lambda c: fetch(c, query_cfg, cache), clusters))
        assignment, components = split_sky_areas(clusters, downloads, cfg['seed'])
        offsets = views()
        split = dict(method='Whole cluster sky areas; merge centers within 6 arcmin or Gaia cones sharing source IDs',
                     seed=cfg['seed'], components=[[clusters[i]['name'] for i in group] for group in components],
                     clusters={label: [clusters[i]['name'] for i in range(len(clusters)) if assignment[i] == label]
                               for label in ('train', 'validation', 'test')},
                     files={label: [] for label in ('train', 'validation', 'test')})
        for view_index in range(len(offsets)):
            for i, cluster in enumerate(clusters):
                stem = f'{i:03d}_{cluster["name"]}' + (f'_v{view_index:02d}' if view_index else '')
                split['files'][assignment[i]].append('sequence_' + stem + '.npz')
        write_json(args.out/'split_5000.json', split)
        print('Sky-area split:', {k: len(v) for k,v in split['clusters'].items()}, flush=True)
        manifest_path = args.out/('manifest_test.json' if partial else 'manifest_5000.json')
        manifest = dict(config=cfg, catalog=str(args.catalog.resolve()),
                        description='100 Gaia clusters, 50 distinct pointings each',
                        offsets_arcsec=offsets, records=[])
        old_path = args.out/'manifest.json'
        old_manifest = json.loads(old_path.read_text(encoding='utf-8')) if old_path.exists() else {'records': []}
        previous = {r.get('file', r.get('fits')): r for r in old_manifest['records']}
        completed = 0
        total = args.limit_sequences or (sum(label == args.split for label in assignment.values()) * 50 if args.split else 5000)
        for view_index, (dx, dy) in enumerate(offsets):
            for cluster_index, (cluster, path) in enumerate(downloads):
                if args.split and assignment[cluster_index] != args.split:
                    continue
                if completed >= total:
                    break
                idx = view_index*len(clusters)+cluster_index
                name = f'{cluster_index:03d}_{cluster["name"]}'
                if view_index:
                    name += f'_v{view_index:02d}'
                output = args.out/(name+'.fits')
                destination = args.out/('sequence_'+name+'.npz')
                png = args.out/(name+'.png')
                local = dict(cfg, seed=cfg['seed']+idx, n_sequences=1,
                             cluster=cluster, catalog='Gaia DR3', membership_selection=False)
                if view_index:
                    center = SkyCoord(cluster['ra']*u.deg, cluster['dec']*u.deg)
                    shifted = center.spherical_offsets_by(dx*u.arcsec, dy*u.arcsec)
                    field = dict(cluster, ra=shifted.ra.deg, dec=shifted.dec.deg)
                    local['field_offset_arcsec'] = [dx, dy]
                else:
                    field = cluster
                actual_offset = [dx, dy]
                if output.exists():
                    with fits.open(output, checksum=True) as h:
                        saved = json.loads(h['CONFIG'].data['JSON'][0])
                        if any(saved.get(k) != v for k, v in local.items() if k != 'field_offset_arcsec'):
                            raise ValueError(f'Existing FITS configuration differs: {output}')
                        actual_offset = saved.get('field_offset_arcsec', actual_offset)
                        if h[0].data.shape != (cfg['n_frames'], cfg['image_size'], cfg['image_size']):
                            raise ValueError(f'Existing FITS shape differs: {output}')
                    if not destination.exists():
                        convert(output, destination)
                    sources = len(fits.getdata(output, 'SOURCES'))
                    diagnostics = previous.get(output.name, {}).get('temporal_check')
                else:
                    if shutil.disk_usage(args.out).free < 10*1024**3:
                        raise OSError('Less than 10 GiB free; stopping')
                    table = Table.read(path, format='ascii.csv')
                    stars, selected, wcs = project(table, field, cfg)
                    if not len(stars):
                        center = SkyCoord(cluster['ra']*u.deg, cluster['dec']*u.deg)
                        offset_rng = np.random.default_rng(local['seed'])
                        for _ in range(500):
                            ax, ay = offset_rng.uniform(-96, 96, 2)
                            shifted = center.spherical_offsets_by(ax*u.arcsec, ay*u.arcsec)
                            field = dict(cluster, ra=shifted.ra.deg, dec=shifted.dec.deg)
                            stars, selected, wcs = project(table, field, cfg)
                            if len(stars):
                                actual_offset = [float(ax), float(ay)]
                                local['field_offset_arcsec'] = actual_offset
                                break
                        else:
                            raise ValueError(f'No Gaia stars around {cluster["name"]}')
                    rng = np.random.default_rng(local['seed'])
                    renderer = Renderer(local, rng, args.device)
                    video, truth = renderer.simulate(stars, rng)
                    diagnostics = renderer.diagnostics
                    local['temporal_quadrature_check'] = diagnostics
                    local['realized_layers'] = [dict(height_m=l['height'], r0_m=float(l['r0']),
                        speed_mps=float(l['speed']), direction_rad=float(l['theta']),
                        screen_shape=list(l['screen'].shape[-2:])) for l in renderer.layers]
                    temporary = output.with_suffix('.partial.fits')
                    save_sequence_fits(temporary, video, truth, stars, local)
                    with fits.open(temporary, mode='update') as h:
                        h[0].header.update(wcs.to_header())
                        h['GROUNDTRUTH'].header.update(wcs.to_header())
                        for key, value in dict(EXPTIME=cfg['exposure_s'], OVERSAMP=cfg['spatial_oversampling'],
                            NSUBEXP=cfg['exposure_substeps'], OBJECT=cluster['name'], FILTER='R_APPROX',
                            CTYPE3='TIME', CUNIT3='s', CRPIX3=1, CRVAL3=cfg['exposure_s']/2,
                            CDELT3=1/cfg['frame_rate_hz']).items():
                            h[0].header[key] = value
                        h.append(fits.BinTableHDU(selected, name='GAIA'))
                        for hdu in h: hdu.add_checksum()
                    temporary.replace(output)
                    convert(output, destination)
                    sources = len(stars)
                    del renderer, video, truth
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                if not png.exists():
                    preview(output, png)
                manifest['records'].append(dict(file=output.name, npz=destination.name,
                    png=png.name, split=assignment[cluster_index],
                    cluster=cluster['name'], view=view_index, offset_arcsec=actual_offset,
                    sources=sources, seed=local['seed'], temporal_check=diagnostics))
                completed += 1
                write_json(manifest_path, manifest)
                write_json(status_path, dict(phase='generating', completed=completed,
                    total=total, last_file=output.name, pid=os.getpid(), updated=time.time()))
                print(f'{completed}/{total} {output.name}: {sources} Gaia stars', flush=True)
            if completed >= total:
                break
        write_json(status_path, dict(phase='complete', completed=completed, total=total,
                                    pid=os.getpid(), updated=time.time()))
    except BaseException as exc:
        write_json(status_path, dict(phase='failed', error=repr(exc), pid=os.getpid(), updated=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
