"""Simulate an unmodulated bead slide, estimate its drift and pupil, and save psf.zip."""
from pathlib import Path

import numpy as np
import tifffile
import torch

from torch_psf import DenseBeadEstimator, VectorialPSF
from torch_psf.psfzip import save_zip


@torch.no_grad()
def simulate(seed=17, size=160, frames=24, beads=260, field_waves=.07):
    rng = np.random.default_rng(seed)
    optics = dict(pixel_um=.10833, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=.7)
    cfg = dict(optics, diameter_um=.04, focus_um=.02)
    model = VectorialPSF(19, **optics, npupil=32, oversample=2, zernike_indices=range(5,12),
                         pixel_integration="midpoint", device="cuda" if torch.cuda.is_available() else "cpu")
    T=lambda x: torch.as_tensor(x,dtype=torch.float32,device=model.dev)
    model.zcoef.copy_(T([.04,.09,-.035,.025,.02,-.015,.04]))
    xy=rng.uniform(0,size,(beads,2))
    brightness=rng.uniform(3500,6500,beads)
    walk=rng.standard_normal((frames,3)).cumsum(0)
    drift=(walk-walk.mean(0))/np.ptp(walk,axis=0)*[1.4,1.1,.65]
    y,x=np.mgrid[-9:10,-9:10]
    nodes=T(np.vstack([np.eye(3),-np.eye(3)])*.02*np.sqrt(.6))
    nodes[:,:2]/=optics['pixel_um']
    movie=[]
    for dx,dy,z in drift:
        pos=xy+[dx,dy]
        anchor=pos.round().astype(int)
        u,v=(2*pos/(size-1)-1).T
        offsets=T(np.column_stack([field_waves*u,field_waves*v,np.zeros((beads,5))]))
        xyz=T(np.column_stack([pos-anchor,np.full(beads,.02)]))
        patches=[]
        for coord,c in zip(xyz.split(16),offsets.split(16)):
            psf=model((coord[:,None]+nodes).flatten(0,1),.02+z,zcoef_offset=c.repeat_interleave(6,0))
            patches.append(psf.reshape(-1,6,19,19).mean(1).cpu().numpy())
        xx=anchor[:,0,None,None]+x
        yy=anchor[:,1,None,None]+y
        ok=(xx>=0)&(xx<size)&(yy>=0)&(yy<size)
        image=np.full((size,size),8.)
        np.add.at(image,(yy[ok],xx[ok]),(np.concatenate(patches)*brightness[:,None,None])[ok])
        movie.append(rng.poisson(image).astype(np.float32))
    return np.stack(movie), cfg, model, drift, xy


def main():
    import matplotlib.pyplot as plt
    path=Path("dense-movie-example")
    path.mkdir(exist_ok=True)
    movie,optics,_,truth,_=simulate()
    tifffile.imwrite(path/"beads.tif",movie,photometric="minisblack",metadata={"axes":"TYX"})
    fit=DenseBeadEstimator.fit_movie(movie,optics=optics,field=True)
    print(fit.field_report)
    if fit.optical_psf is not None:
        save_zip(path/"psf.zip",fit.optical_psf)
    np.savez(path/"beads.npz",xy=fit.xy,photons=fit.photons,drift_px=fit.drift,axial_drift_um=fit.axial_drift_um)
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout="constrained")
    axes[0].imshow(movie[0],cmap="gray",interpolation="nearest",vmax=np.percentile(movie[0],99.7))
    axes[0].scatter(*(fit.xy+fit.drift[0]).T,s=4,facecolors="none",edgecolors="cyan",linewidths=.4)
    axes[0].set_title("Detected beads, frame 1")
    axes[1].plot(truth[:,2]*1000,label="True")
    if fit.axial_drift_um is not None:
        axes[1].plot((fit.axial_drift_um-fit.axial_drift_um.mean())*1000,label="Estimated")
    axes[1].set(xlabel="Frame",ylabel="Relative focus (nm)");axes[1].legend()
    if fit.field_report["selected"]=="field":
        yy,xx=np.mgrid[0:movie.shape[1]:32j,0:movie.shape[2]:32j]
        maps=fit.optical_psf.zernikes_at(np.column_stack([xx.ravel(),yy.ravel()]))
        im=axes[2].imshow(maps[:,0].reshape(xx.shape)*1000,extent=[0,xx.max(),yy.max(),0])
        fig.colorbar(im,ax=axes[2],label="mλ RMS")
    axes[2].set_title("Fitted Z5" if fit.field_report["selected"]=="field" else "Global pupil selected")
    fig.savefig(path/"field.png",dpi=140)


if __name__=="__main__":
    main()
