"""Convert simulator FITS files to sequence_*.npz files for dancecam.train."""
import argparse
import json
from pathlib import Path

from astropy.io import fits
import numpy as np


def convert(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    with fits.open(source) as hdus:
        video = np.asarray(hdus[0].data, dtype=np.float32)
        target = np.asarray(hdus['GROUNDTRUTH'].data, dtype=np.float32)
        raw = hdus['CONFIG'].data['JSON'][0]
        config_json = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
        config = json.loads(config_json)
        assert video.shape == (config['n_frames'], config['image_size'], config['image_size'])
        assert target.shape == video.shape[1:]
        catalog = np.column_stack([hdus['SOURCES'].data[key]
                                   for key in ('X', 'Y', 'MAG', 'FLUX')]).astype(np.float32)
        payload = dict(video=video, target=target, catalog=catalog,
                       config_json=config_json, source_fits=source.name,
                       fits_header=hdus[0].header.tostring())
        if 'GAIA' in hdus:
            payload['gaia'] = np.array(hdus['GAIA'].data)
        temporary = destination.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **payload)
        # Verify lossless image conversion and the training-required metadata.
        with np.load(temporary, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['video'], video)
            np.testing.assert_array_equal(data['target'], target)
            assert json.loads(str(data['config_json'])) == config
            np.testing.assert_array_equal(data['catalog'], catalog)
            if 'gaia' in payload:
                np.testing.assert_array_equal(data['gaia'], payload['gaia'])
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(p for p in args.directory.glob('*.fits') if not p.name.endswith('.partial.fits'))
    if not files:
        parser.error('No FITS files found')
    destinations = [p.with_name('sequence_' + p.stem + '.npz') for p in files]
    existing = [p for p in destinations if p.exists()]
    if existing:
        parser.error(f'Output already exists: {existing[0]}')
    for idx, (source, destination) in enumerate(zip(files, destinations), 1):
        convert(source, destination)
        print(f'{idx}/{len(files)} {destination.name}', flush=True)


if __name__ == '__main__':
    main()
