"""Local phase diversity for a stationary, uniformly illuminated planar bead movie."""
import numpy as np
import torch

from ..field import FieldPSF, FieldZernikes
from .bead_movie import FiniteBeadRenderer


@torch.no_grad()
def fit_movie_field(frames, drift_px, focus_um, optics):
    """Fit affine Noll fields with a shared focus trace; hold out exposures for selection.

    Each tapered image patch has an independent unknown object spectrum. Profile
    it on training exposures and predict the reserved exposures at their focus.
    Overlapping bead images enter as one object; bead intensities must be stable.
    """
    base = optics["calibrated_psf"]
    times = np.rint(np.linspace(0, len(frames)-1, min(12, len(frames)))).astype(int)
    h, w = frames.shape[1:]
    if len(times) < 6 or min(h, w) < 64 or np.ptp(focus_um[times]) < .04:
        return base, {"selected": "global", "reason": "Need six exposures, a 64 px field and 40 nm of focus diversity"}
    renderer = FiniteBeadRenderer(dict(optics), 9)
    T = renderer.tensor
    z = T(focus_um[times])
    n = len(z)
    side = min(64, 2 ** int(np.floor(np.log2(min(h, w)/2))))
    xy = np.array([(x, y) for y in np.rint(np.linspace(0,h-side,3)).astype(int)
                   for x in np.rint(np.linspace(0,w-side,3)).astype(int)])
    center = xy + (side-1)/2
    spatial = T(np.column_stack([np.ones(len(xy)), 2*center[:,1]/(h-1)-1, 2*center[:,0]/(w-1)-1]))
    freq = torch.fft.fftfreq(side, device=z.device)
    qy, qx = torch.meshgrid(freq, freq, indexing="ij")
    band = (qy >= 0) & (qx.hypot(qy) >= .05) & (qx.hypot(qy) <= .25)
    win = torch.hann_window(side, periodic=False, device=z.device)
    win = win[:,None]*win
    images = T(frames[times])
    patches = torch.stack([images[:,y:y+side,x:x+side] for x,y in xy])
    mean = (patches*win).sum((-2,-1),keepdim=True)/win.sum()
    data = torch.fft.fft2((patches-mean)*win)[...,band]
    shifts = T(drift_px[times])
    data *= torch.exp(2j*torch.pi*(shifts[:,0,None]*qx[band]+shifts[:,1,None]*qy[band]))
    train = torch.arange(n, device=z.device)%4 != 2
    power = data[:,train].abs().square().mean(1, keepdim=True)
    data /= (power + power.mean(-1,keepdim=True)*.15).clamp_min(1).sqrt()
    modes = len(base.zidx)
    nodes = renderer.nodes.clone()
    nodes[:,2] += optics["diameter_um"]/2
    def otf(coef):
        pos = nodes.repeat(n,1)
        stamps = renderer.model(pos, z.repeat_interleave(6), zcoef_offset=coef.expand(len(pos),-1))
        stamps = stamps.reshape(n,6,19,19).mean(1)
        return torch.fft.fft2(stamps, s=(side,side))[:,band]
    def residual(H, Y, mask):
        htrain = H[train]
        obj = (htrain.conj()*Y[train]).sum(0) / htrain.abs().square().sum(0).clamp_min(1e-10)
        return (H*obj-Y)[mask]
    def system(coeff, X, jac=False, held=False):
        size = coeff.numel()
        gram = z.new_zeros(size,size)
        rhs = z.new_zeros(size)
        cost = z.new_zeros(())
        for v,Y in zip(X,data):
            pupil = v@coeff
            H = otf(pupil)
            r = residual(H, Y, ~train if held else train)
            cost += r.abs().square().sum()
            if jac:
                cols=[]
                for direction in torch.eye(modes, device=z.device):
                    d = (residual(otf(pupil+.001*direction),Y,train)-r)/.001
                    cols.append(torch.view_as_real(d).flatten())
                J=(v[:,None,None]*torch.stack(cols)[None]).reshape(size,-1)
                gram += J@J.T
                rhs -= J@torch.view_as_real(r).flatten()
        return float(cost),gram,rhs
    def optimise(coeff,X):
        for _ in range(5):
            cost,gram,rhs=system(coeff,X,True)
            gram.diagonal().add_(.02*gram.diagonal().clone()+1e-4)
            step=torch.linalg.solve(gram,rhs).reshape_as(coeff)
            step /= (step.abs().max()/.045).clamp_min(1)
            for scale in (1,.5,.25):
                trial=coeff+scale*step
                if system(trial,X)[0] < cost:
                    coeff=trial
                    break
            else:
                break
        return coeff
    constant = optimise(z.new_zeros(1,modes),spatial[:,:1])
    affine = optimise(torch.cat([constant,z.new_zeros(2,modes)]),spatial)
    nominal = system(torch.zeros_like(constant),spatial[:,:1],held=True)[0]
    shared = system(constant,spatial[:,:1],held=True)[0]
    varied = system(affine,spatial,held=True)[0]
    den = len(xy)*int((~train).sum())*int(band.sum())
    report = dict(global_score=min(nominal,shared)/den, field_score=varied/den,
                  held_frames=times[(~train).cpu().numpy()].tolist(), selected="global")
    if varied >= .99*min(nominal,shared) or not np.isfinite(varied):
        return base, report
    center_psf=base.resized(base.roisize)
    center_psf.zcoef.add_(affine[0])
    norm=((w-1)/2,(h-1)/2,(w-1)/2,(h-1)/2)
    field=FieldZernikes(base.zidx,order=1,norm=norm,coef=affine[1:].T.cpu().numpy(),device=z.device)
    report["selected"]="field"
    return FieldPSF(center_psf,field),report


@torch.no_grad()
def field_photometry(movie, fit, model, optics):
    """Allocate raw pixel photons jointly to local finite-bead PSFs at the fitted coordinates."""
    renderer=FiniteBeadRenderer(dict(optics,calibrated_psf=model.base),fit.psf.radius)
    T=renderer.tensor
    h,w=movie.shape[1:]
    axis=torch.arange(-fit.psf.radius,fit.psf.radius+1,device=model.dev)
    dy,dx=torch.meshgrid(axis,axis,indexing="ij")
    for t,image in enumerate(movie):
        pos=T(fit.xy+fit.drift[t])
        anchor=pos.round().long()
        x=anchor[:,0,None]+dx.flatten()
        y=anchor[:,1,None]+dy.flatten()
        valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        index=(y*w+x).clamp(0,h*w-1)
        offset=pos-anchor
        xyz=torch.cat([offset,torch.full_like(offset[:,:1],optics["diameter_um"]/2)],1)
        pupil=model.offsets(pos)
        kernels=[]
        for coords,coef in zip(xyz.split(16),pupil.split(16)):
            nodes=(coords[:,None]+renderer.nodes).flatten(0,1)
            stamps=renderer.model(nodes,float(optics["focus_um"]+fit.axial_drift_um[t]),
                                  zcoef_offset=coef.repeat_interleave(6,0))
            kernels.append(stamps.reshape(len(coords),6,-1).mean(1))
        kernel=torch.cat(kernels)*valid
        b=T(fit.background[t])
        bg=b[0]+b[1]*torch.linspace(-1,1,w,device=model.dev)[None,:]+b[2]*torch.linspace(-1,1,h,device=model.dev)[:,None]
        flux=T(fit.photons[t])
        obs=T(image).flatten()
        for _ in range(8):
            mu=bg.flatten().scatter_add(0,index.flatten(),(kernel*flux[:,None]).flatten()).clamp_min(.01)
            flux=flux*((obs/mu)[index]*kernel).sum(1)/kernel.sum(1).clamp_min(1e-8)
        fit.photons[t]=flux.cpu().numpy()
