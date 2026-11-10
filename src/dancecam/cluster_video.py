"""Generate cluster-centred videos from cached Gaia DR3 sky-field sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen, Request

import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

from dancecam.simulate import simulate_one, save_sequence_fits

ENDPOINT = 'https://gaia.ari.uni-heidelberg.de/tap/sync'


def read_clusters(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.strip().strip('|').split('|')
        if len(fields) == 9 and fields[0].isdigit():
            rows.append(dict(name=fields[1], cluster_id=int(fields[2]),
                             ra=float(fields[5]), dec=float(fields[6])))
    return rows


def fetch(cluster, cfg, cache):
    path = cache / (cluster['name'] + '.csv')
    radius = np.sqrt(2) * cfg['image_size'] * cfg['pixel_scale_arcsec'] / 7200 * 1.01
    query = ("SELECT source_id,ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source "
             f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{cluster['ra']},"
             f"{cluster['dec']},{radius})) AND phot_g_mean_mag BETWEEN "
             f"{cfg['magnitude_min']} AND {cfg['magnitude_max']} ORDER BY source_id")
    if not path.exists():
        payload = urlencode(dict(REQUEST='doQuery', LANG='ADQL', FORMAT='csv',
                                 QUERY=query, MAXREC=100000)).encode()
        for attempt in range(3):
            try:
                with urlopen(Request(ENDPOINT, data=payload), timeout=180) as response:
                    content = response.read().decode('utf-8')
                table = Table.read(content.splitlines(), format='ascii.csv')
                if not {'source_id', 'ra', 'dec', 'phot_g_mean_mag'} <= set(table.colnames):
                    raise ValueError(content[:500])
                if len(table) >= 100000:
                    raise ValueError('Possible truncated Gaia response')
                path.write_text(content, encoding='utf-8')
                path.with_suffix('.adql').write_text(query, encoding='utf-8')
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    return cluster, path


def project(table, cluster, cfg):
    wcs = WCS(naxis=2)
    size = cfg['image_size']
    wcs.wcs.crpix = [(size + 1) / 2] * 2
    wcs.wcs.cdelt = [-cfg['pixel_scale_arcsec'] / 3600, cfg['pixel_scale_arcsec'] / 3600]
    wcs.wcs.crval = [cluster['ra'], cluster['dec']]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    x, y = wcs.world_to_pixel_values(table['ra'], table['dec'])
    keep = (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
    selected = table[keep]
    mag = np.asarray(selected['phot_g_mean_mag'], dtype=float)
    flux = cfg['electrons_at_mag15'] * 10 ** (-0.4 * (mag - 15))
    return np.column_stack((x[keep], y[keep], mag, flux)), selected, wcs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/paper_approx.json'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=50)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    clusters = read_clusters(args.catalog)[:args.count]
    if len(clusters) != args.count:
        parser.error('Not enough cluster rows')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / 'gaia_cache'
    cache.mkdir(exist_ok=True)
    # Download concurrently; simulation remains sequential with independent seeds.
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(lambda c: fetch(c, cfg, cache), clusters))
    print(f'Gaia queries complete: {len(downloads)} fields', flush=True)
    manifest = []
    for idx, (cluster, path) in enumerate(downloads):
        table = Table.read(path, format='ascii.csv')
        stars, selected, wcs = project(table, cluster, cfg)
        if not len(stars):
            raise ValueError(f'No Gaia sources in field: {cluster}')
        local = dict(cfg, seed=cfg['seed'] + idx, n_sequences=1,
                     cluster=cluster, catalog='Gaia DR3', gaia_endpoint=ENDPOINT,
                     magnitude_band='Gaia G', membership_selection=False,
                     flux_calibration='Approximate electrons_at_mag15 using Gaia G',
                     source_count=len(stars))
        output = args.out_dir / f'{idx:03d}_{cluster["name"]}.fits'
        if not output.exists():
            video, target, stars = simulate_one(local, np.random.default_rng(local['seed']), stars)
            temporary = output.with_suffix('.partial.fits')
            save_sequence_fits(temporary, video, target, stars, local)
            with fits.open(temporary, mode='update') as hdus:
                hdus[0].header.update(wcs.to_header())
                hdus[0].header['CTYPE3'] = 'TIME'
                hdus[0].header['CUNIT3'] = 's'
                hdus[0].header['CRPIX3'] = 1
                hdus[0].header['CRVAL3'] = 0
                hdus[0].header['CDELT3'] = 1 / cfg['frame_rate_hz']
                hdus[0].header['OBJECT'] = cluster['name']
                hdus[0].header['CATALOG'] = 'Gaia DR3'
                hdus[0].header['MAGBAND'] = 'Gaia G'
                hdus['GROUNDTRUTH'].header.update(wcs.to_header())
                hdus.append(fits.BinTableHDU(selected, name='GAIA'))
                for hdu in hdus:
                    hdu.add_checksum()
            temporary.replace(output)
        manifest.append(dict(file=output.name, **cluster, sources=len(stars), seed=local['seed']))
        (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'{idx+1}/{len(clusters)} {cluster["name"]}: {len(stars)} Gaia sources', flush=True)


if __name__ == '__main__':
    main()
