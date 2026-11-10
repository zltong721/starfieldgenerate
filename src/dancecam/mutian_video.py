"""GPU optical propagation, subpixel imaging and exposure integration for MOST proxy data."""
import argparse
import json
from pathlib import Path
import time

from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy.fft import next_fast_len
import torch
from torch.nn import functional as F

from .cluster_video import project, read_clusters
from .fits_to_npz import convert
from .simulate import ARCSEC_TO_RAD, phase_screen_rect, save_sequence_fits


class Renderer:
    def __init__(self, cfg, rng, device='cuda'):
        self.cfg, self.device = cfg, torch.device(device)
        self.pitch = cfg['aperture_m'] / cfg['pupil_grid']
        n = cfg['propagation_grid']
        axis = (torch.arange(n, device=device) - (n-1)/2) * self.pitch
        self.yy, self.xx = torch.meshgrid(axis, axis, indexing='ij')
        radius = torch.hypot(self.xx, self.yy)
        self.pupil = ((radius <= cfg['aperture_m']/2) &
                      (radius >= cfg['obstruction_diameter_m']/2)).float()
        inner, outer = .75*cfg['aperture_m'], cfg['aperture_m']
        self.window = torch.where(radius <= inner, 1., torch.where(radius >= outer, 0.,
                       .5*(1+torch.cos(torch.pi*(radius-inner)/(outer-inner)))))
        freq = torch.fft.fftfreq(n, self.pitch, device=device)
        self.f2 = freq[:, None]**2 + freq[None, :]**2
        self.layers = []
        duration = (cfg['n_frames']-1)/cfg['frame_rate_hz'] + cfg['exposure_s']
        half_extent = n*self.pitch/np.sqrt(2)
        max_angle = cfg['image_size']*cfg['pixel_scale_arcsec']*ARCSEC_TO_RAD/2
        for height, rmean, rsigma, vmean, vsigma in zip(
                cfg['layer_heights_m'], cfg['layer_r0_mean_m'], cfg['layer_r0_sigma_m'],
                cfg['layer_wind_mean_mps'], cfg['layer_wind_sigma_mps']):
            r0 = max(.03, rng.normal(rmean, rsigma))
            speed = max(0., rng.normal(vmean, vsigma))
            theta = rng.uniform(0, 2*np.pi)
            margin = half_extent + np.sqrt(2)*height*max_angle + 4*self.pitch
            nx = next_fast_len(int(np.ceil((speed*duration + 2*margin)/self.pitch)))
            ny = next_fast_len(int(np.ceil(2*margin/self.pitch)))
            screen = phase_screen_rect(nx, ny, self.pitch, r0, cfg['outer_scale_m'],
                                       cfg['inner_scale_m'], rng)
            self.layers.append(dict(screen=torch.from_numpy(screen).to(device)[None, None],
                               speed=speed, theta=theta, height=height, r0=r0, margin=margin))
        factor = cfg['spatial_oversampling']
        self.kernel_size = cfg['psf_stamp']*factor + 1
        k = self.kernel_size
        arcsec = (torch.arange(k, device=device)-(k-1)/2)*cfg['pixel_scale_arcsec']/factor
        native_scale = cfg['wavelength_nm']*1e-9/(n*self.pitch)/ARCSEC_TO_RAD
        index = arcsec/native_scale + n//2
        gy, gx = torch.meshgrid(2*index/(n-1)-1, 2*index/(n-1)-1, indexing='ij')
        self.psf_grid = torch.stack([gx, gy], dim=-1)[None]
        self.diagnostics = []

    def kernels(self, times, angle, ideal=False):
        """Return normalized high resolution instantaneous PSFs."""
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        batch = len(times)
        field = (self.pupil if ideal else self.window).to(torch.complex64)[None].repeat(batch,1,1)
        if not ideal:
            for i, layer in enumerate(self.layers):
                x = self.xx + layer['height']*angle[0]
                y = self.yy + layer['height']*angle[1]
                ct, st = np.cos(layer['theta']), np.sin(layer['theta'])
                along = x*ct + y*st
                across = -x*st + y*ct
                screen = layer['screen']
                ny, nx = screen.shape[-2:]
                sx = (along[None] + layer['speed']*times[:,None,None] + layer['margin'])/self.pitch
                sy = across[None]/self.pitch + (ny-1)/2
                grid = torch.stack([2*sx/(nx-1)-1, (2*sy/(ny-1)-1).expand(batch,-1,-1)],dim=-1)
                phase = F.grid_sample(screen.expand(batch,-1,-1,-1), grid,
                                      align_corners=True, padding_mode='zeros')[:,0]
                field *= torch.exp(1j*phase)
                if i+1 < len(self.layers):
                    dz = layer['height']-self.layers[i+1]['height']
                    if dz > 0:
                        transfer = torch.exp(-1j*torch.pi*self.cfg['wavelength_nm']*1e-9*dz*self.f2)
                        field = torch.fft.ifft2(torch.fft.fft2(field)*transfer)
            field *= self.pupil
        intensity = torch.fft.fftshift(torch.fft.fft2(field).abs().square(),dim=(-2,-1))
        kernel = F.grid_sample(intensity[:,None], self.psf_grid.expand(batch,-1,-1,-1),
                               align_corners=True)[:,0].clamp(min=0)
        return kernel/kernel.sum(dim=(-2,-1),keepdim=True)

    def exposure_kernel(self, start, angle, steps=None):
        steps = steps or self.cfg['exposure_substeps']
        times = start+(np.arange(steps)+.5)*self.cfg['exposure_s']/steps
        result = torch.zeros((self.kernel_size,self.kernel_size),device=self.device)
        # Bound memory during padded Fresnel propagation.
        for offset in range(0,steps,4):
            result += self.kernels(times[offset:offset+4],angle).sum(dim=0)
        return result/steps

    def simulate(self, stars, rng):
        cfg = self.cfg
        size, factor = cfg['image_size'], cfg['spatial_oversampling']
        fine_size = size*factor
        full = next_fast_len(fine_size+self.kernel_size-1)
        half = self.kernel_size//2
        zone_count = 2 if size >=64 else 1
        zones = []
        for zy in range(zone_count):
            for zx in range(zone_count):
                sub = stars[(stars[:,0]>=zx*size/zone_count)&(stars[:,0]<(zx+1)*size/zone_count)&
                            (stars[:,1]>=zy*size/zone_count)&(stars[:,1]<(zy+1)*size/zone_count)]
                scene = np.zeros((fine_size,fine_size),dtype=np.float32)
                for x,y,_,flux in sub:
                    # Keep detector pixel centers aligned after block integration.
                    px,py=x*factor+(factor-1)/2,y*factor+(factor-1)/2
                    ix,iy=int(px),int(py)
                    dx,dy=px-ix,py-iy
                    for ox,oy,w in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                        if 0<=ix+ox<fine_size and 0<=iy+oy<fine_size:
                            scene[iy+oy,ix+ox]+=flux*w
                angle=((zx+.5-zone_count/2)*size/zone_count*cfg['pixel_scale_arcsec']*ARCSEC_TO_RAD,
                       (zy+.5-zone_count/2)*size/zone_count*cfg['pixel_scale_arcsec']*ARCSEC_TO_RAD)
                spectrum=torch.fft.rfft2(torch.from_numpy(scene).to(self.device),s=(full,full))
                zones.append((spectrum,angle))

        def integrate(spectrum):
            fine=torch.fft.irfft2(spectrum,s=(full,full))[half:half+fine_size,half:half+fine_size]
            return fine.reshape(size,factor,size,factor).sum(dim=(1,3)).clamp(min=0).cpu().numpy()

        ideal=self.kernels([0.],(0.,0.),ideal=True)[0]
        truth=integrate(sum(s for s,a in zones)*torch.fft.rfft2(ideal,s=(full,full)))
        # Probe temporal quadrature at one representative line of sight and exposure.
        a=zones[0][1]
        coarse=self.exposure_kernel(0.,a)
        fine=self.exposure_kernel(0.,a,2*cfg['exposure_substeps'])
        self.diagnostics.append(dict(exposure_steps=cfg['exposure_substeps'],
                    doubled_steps=2*cfg['exposure_substeps'],
                    kernel_relative_l2=float(torch.linalg.vector_norm(coarse-fine)/torch.linalg.vector_norm(fine))))
        video=np.empty((cfg['n_frames'],size,size),dtype=np.float32)
        for frame in range(cfg['n_frames']):
            combined=None
            for spectrum,angle in zones:
                kernel=self.exposure_kernel(frame/cfg['frame_rate_hz'],angle)
                contribution=spectrum*torch.fft.rfft2(kernel,s=(full,full))
                combined=contribution if combined is None else combined+contribution
            signal=integrate(combined)
            noisy=rng.poisson(signal+cfg['sky_e_per_pixel']).astype(np.float32)
            noisy+=rng.normal(0,cfg['read_noise_e'],noisy.shape).astype(np.float32)
            video[frame]=noisy
            if (frame+1)%16==0:
                print(f'  frame {frame+1}/{cfg["n_frames"]}',flush=True)
        return video,truth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/mutian_193_r.json'))
    parser.add_argument('--catalog',type=Path,default=Path('data/source/catalogs/Hunt_Reffert_2024_100_candidates.md'))
    parser.add_argument('--cache',type=Path,default=Path('data/generated/starcluster/gaia_cache'))
    parser.add_argument('--out',type=Path,default=Path('data/generated/mutian_193_r'))
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    started=time.monotonic()
    for idx,cluster in enumerate(read_clusters(args.catalog)[:args.count]):
        local=dict(cfg,seed=cfg['seed']+idx,n_sequences=1,cluster=cluster,catalog='Gaia DR3',membership_selection=False)
        output=args.out/f'{idx:03d}_{cluster["name"]}.fits'
        if output.exists():
            raise FileExistsError(f'Use a fresh output directory: {output}')
        table=Table.read(args.cache/(cluster['name']+'.csv'),format='ascii.csv')
        stars,selected,wcs=project(table,cluster,local)
        print(f'{idx+1}/{args.count} {cluster["name"]}: {len(stars)} stars',flush=True)
        rng=np.random.default_rng(local['seed'])
        renderer=Renderer(local,rng,args.device)
        video,truth=renderer.simulate(stars,rng)
        local['temporal_quadrature_check']=renderer.diagnostics
        local['realized_layers']=[dict(height_m=l['height'],r0_m=float(l['r0']),speed_mps=float(l['speed']),
                                 direction_rad=float(l['theta']),screen_shape=list(l['screen'].shape[-2:])) for l in renderer.layers]
        temporary=output.with_suffix('.partial.fits')
        save_sequence_fits(temporary,video,truth,stars,local)
        with fits.open(temporary,mode='update') as h:
            h[0].header.update(wcs.to_header())
            h['GROUNDTRUTH'].header.update(wcs.to_header())
            for key,value in dict(EXPTIME=cfg['exposure_s'],OVERSAMP=cfg['spatial_oversampling'],
                                  NSUBEXP=cfg['exposure_substeps'],OBJECT=cluster['name'],FILTER='R_APPROX',
                                  CTYPE3='TIME',CUNIT3='s',CRPIX3=1,CRVAL3=cfg['exposure_s']/2,
                                  CDELT3=1/cfg['frame_rate_hz']).items():
                h[0].header[key]=value
            h.append(fits.BinTableHDU(selected,name='GAIA'))
            for hd in h: hd.add_checksum()
        temporary.replace(output)
        destination=output.with_name('sequence_'+output.stem+'.npz')
        convert(output,destination)
        manifest.append(dict(file=output.name,npz=destination.name,cluster=cluster['name'],sources=len(stars),
                             seed=local['seed'],temporal_check=renderer.diagnostics,
                             target_flux=float(truth.sum()),catalog_flux=float(stars[:,3].sum())))
        (args.out/'manifest.json').write_text(json.dumps(dict(config=cfg,records=manifest,
                                            elapsed_s=time.monotonic()-started),indent=2))
        print(f'Saved {output.name}; temporal check {renderer.diagnostics}',flush=True)
        del renderer,video,truth
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=='__main__':
    main()
