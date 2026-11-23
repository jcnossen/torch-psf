"""Global sparse pixel model for overlapping fluorescent beads."""
import numpy as np
from scipy import ndimage, optimize, spatial, sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
import torch
from dataclasses import dataclass
from .dense_beads import find_all_beads


class FiniteBeadRenderer:
    """Pixel-integrated vectorial PSF at six cubature nodes of a uniformly fluorescent sphere."""
    def __init__(self, cfg, radius=7):
        import torch_psf as tp
        self.side = 2*radius+1
        self.cfg = cfg
        self.radius = cfg['diameter_um']/2
        calibrated = cfg.get('calibrated_psf')
        self.model = calibrated.resized(self.side) if calibrated is not None else tp.VectorialPSF(self.side, **{k:cfg[k] for k in
            ('pixel_um','NA','n_med','n_imm','wavelength_um','focus_um')},
            zernike_indices=(6,), npupil=32, oversample=2)
        self.device = self.model.dev
        self.model.requires_grad_(False)
        # Equal-weight six-node cubature integrates second moments of a uniform solid sphere.
        nodes = np.vstack((np.eye(3), -np.eye(3)))*self.radius*np.sqrt(3/5)
        nodes[:, :2] /= cfg['pixel_um']
        self.nodes = self.tensor(nodes)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def render(self, off, aberration):
        n = len(off)
        z = torch.full_like(off[:, :1], self.radius)
        pos = torch.cat((off, z), 1)[:, None, :] + self.nodes
        field = aberration[:, None].expand(n, 6).reshape(-1, 1)
        # Avoid excessively large pupil batches, including on a CPU-only installation.
        images = [self.model(p, self.cfg['focus_um'], zcoef_offset=None if self.cfg.get('calibrated_psf') is not None else a)
                  for p, a in zip(pos.reshape(-1, 3).split(48), field.split(48))]
        return torch.cat(images).reshape(n, 6, self.side, self.side)


class PixelModel:
    """Columns are finite-bead vectorial PSFs. The final three columns form a background plane."""
    def __init__(self, optics, shape):
        self.shape = shape
        self.radius = optics.model.roisize // 2
        grid = np.linspace(-.52, .52, 7)
        levels = np.linspace(-.18, .18, 25)
        z, y, x = np.meshgrid(levels, grid, grid, indexing='ij')
        offsets = np.stack((x.ravel(), y.ravel()), axis=1)
        images = []
        with torch.no_grad():
            for i in range(0, len(offsets), 32):
                kernels = optics.render(optics.tensor(offsets[i:i+32]), optics.tensor(z.ravel()[i:i+32]))
                images.extend(kernels.mean(1).cpu().numpy())
        self.table = RegularGridInterpolator((levels, grid, grid), np.array(images).reshape(*z.shape, *images[0].shape))
        self.init_grid(shape, self.radius)

    def init_grid(self, shape, radius):
        self.shape, self.radius = shape, radius
        side = np.arange(-self.radius, self.radius+1)
        self.dy, self.dx = np.meshgrid(side, side, indexing='ij')
        yy, xx = np.indices(shape)
        self.plane = sparse.csc_matrix(self.field_basis(np.c_[xx.ravel(), yy.ravel()]))

    def field_basis(self, xy):
        return np.column_stack((np.ones(len(xy)), 2*xy/(np.array(self.shape[::-1])-1)-1))

    def kernels(self, xy, args):
        return self.table(args)

    def columns(self, xy, coeff, derivatives=False):
        anchor = np.rint(xy).astype(int)
        x = anchor[:, 0, None, None]+self.dx
        y = anchor[:, 1, None, None]+self.dy
        h,w = self.shape
        valid = (x>=0)&(x<w)&(y>=0)&(y<h)
        row = (y*w+x)[valid]
        col = np.broadcast_to(np.arange(len(xy))[:,None,None], valid.shape)[valid]
        off = xy-anchor
        args = np.column_stack((np.clip(self.field_basis(xy)@coeff,-.179,.179),off[:,1],off[:,0]))
        def matrix(vals):
            return sparse.csc_matrix((vals[valid], (row,col)), shape=(h*w,len(xy)))
        base = matrix(self.kernels(xy,args))
        if not derivatives:
            return base
        delta = [.001,.008,.008]
        deriv = []
        for i in (2,1,0):
            step = np.zeros(3)
            step[i] = delta[i]
            deriv.append(matrix((self.kernels(xy,args+step)-self.kernels(xy,args-step))/(2*delta[i])))
        return base, *deriv


def solve_normal(gram, rhs, ridge=1e-7):
    # Diagonal scaling keeps coordinates in pixels, photons and waves well conditioned.
    scale = 1/np.sqrt(np.maximum(gram.diagonal(),1e-12))
    D = sparse.diags(scale)
    scaled = D@gram@D+sparse.eye(len(rhs),format='csc')*ridge
    return scale*spsolve(scaled.tocsc(), scale*rhs)


def fit_linear(A, observed, weight, nbeads):
    """Weighted nonnegative bead flux and unconstrained background."""
    WA = A.multiply(weight[:,None]).tocsr()
    gram = (WA.T@WA).tocsc()
    rhs = np.asarray(WA.T@(observed*weight)).ravel()
    # Active set with KKT re-entry. Background and any motion terms stay free.
    active = np.ones(A.shape[1],dtype=bool)
    for _ in range(30):
        params = np.zeros(A.shape[1])
        params[active] = solve_normal(gram[active][:,active],rhs[active])
        bad = np.where(active[:nbeads] & (params[:nbeads] < -1e-6))[0]
        if len(bad):
            active[bad] = False
            continue
        grad = rhs-gram@params
        missing = np.where(~active[:nbeads] & (grad[:nbeads]>1e-5))[0]
        if not len(missing):
            break
        active[missing[np.argmax(grad[missing])]] = True
    params[:nbeads] = np.maximum(params[:nbeads],0)
    return params


def grouped_images(movie, groups):
    """Average repeated exposure conditions for geometry; preserve every frame for photometry."""
    if groups is None:
        return movie, np.ones(len(movie), dtype=int)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(movie):
        raise ValueError("groups must contain one condition ID per frame")
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    return np.stack([movie[inv == j].mean(0) for j in range(len(counts))]), counts


def propose(score, threshold, xy=None, size=3):
    peaks = (score==ndimage.maximum_filter(score,size=size)) & (score>threshold)
    pts = np.argwhere(peaks)[:, ::-1]
    if xy is not None and len(pts):
        # Avoid re-seeding the identical pixel centre; close neighbours can still enter.
        pts = pts[spatial.cKDTree(xy).query(pts)[0]>.65]
    return pts.astype(float)


def joint_positions(model, images, counts, xy, coeff, iterations):
    """Profile each phase's flux and fit common positions and a Z6 plane to the summed image."""
    n = len(xy)
    if n<1:
        raise ValueError('No bead candidates detected.')
    flat = images.reshape(len(images),-1)
    for iteration in range(iterations+1):
        psf,dx,dy,dz = model.columns(xy,coeff,derivatives=True)
        A = sparse.hstack((psf,model.plane),format='csc')
        basis = model.field_basis(xy)
        blocks, rhs, residuals, fluxes = [],[],[],[]
        for obs, count in zip(flat,counts):
            weight = np.sqrt(count/np.maximum(obs,3))
            amps = fit_linear(A,obs,weight,n)
            flux = amps[:n]
            residual = obs-A@amps
            residuals.append(residual)
            fluxes.append(flux)
            if iteration==iterations:
                continue
            deriv = sparse.hstack((dx.multiply(flux),dy.multiply(flux),
                                   sparse.csc_matrix(dz@(basis*flux[:,None]))),format='csr')
            # Include the flux and background nuisance derivatives in the position update.
            # This lets a close pair trade flux as its fitted separation changes.
            WJ = deriv.multiply(weight[:,None]).tocsr()
            WA = A.multiply(weight[:,None]).tocsr()
            blocks.append((WJ,WA))
            rhs.append(residual*weight)
        if iteration==iterations:
            break
        geom = sparse.vstack([g for g,a in blocks])
        linear = sparse.block_diag([a for g,a in blocks])
        jac = sparse.hstack((geom,linear),format='csr')
        step = solve_normal(jac.T@jac,jac.T@np.concatenate(rhs),ridge=.0008)
        move = np.stack((step[:n],step[n:2*n]),axis=1)
        xy = np.clip(xy+move/np.maximum(1,np.linalg.norm(move,axis=1,keepdims=True)/.5),
                     0,np.array(model.shape[::-1])-1)
        coeff = coeff+np.clip(step[2*n:2*n+3],-.012,.012)
    return xy,coeff,np.array(fluxes),np.array(residuals).reshape(images.shape)


def deblend(model, movie, groups, update_psf=None):
    images,counts = grouped_images(movie, groups)
    xy, _ = find_all_beads(images, sigma=.65, threshold=8., min_dist=1, bg_size=9)
    coeff = np.zeros(3)
    if update_psf is not None:
        for _ in range(4):
            xy,coeff,amps,resid = joint_positions(model,images,counts,xy,coeff,3)
            update_psf(xy,amps,resid,counts)
    for attempt in range(4):
        xy,coeff,amps,resid = joint_positions(model,images,counts,xy,coeff,10 if attempt else 12)
        if attempt==3:
            break
        # A phase-dependent positive residual is evidence for a second PSF, even when
        # its centre was invisible in the mean image.
        smooth = np.stack([ndimage.gaussian_filter(v,.72) for v in resid])
        noise = np.sqrt(np.maximum(images,4)/counts[:,None,None])
        score = (smooth/noise).max(0)
        new = propose(score,5.5,xy)
        if not len(new): break
        xy = np.vstack((xy,new))
    # Remove unsupported extra components. The criterion is a signal threshold,
    # independent of nearest-neighbour distance.
    keep = amps.mean(0)>max(40,.055*np.median(amps.mean(0)))
    if not keep.all():
        xy,coeff,amps,resid = joint_positions(model,images,counts,xy[keep],coeff,7)
    return merge_unsupported_splits(model,images,counts,xy,coeff)


def merge_unsupported_splits(model, images, counts, xy, coeff):
    """Compare close two-source hypotheses with a refitted one-source alternative."""
    nphase = len(images)
    for _ in range(4):
        pairs = spatial.cKDTree(xy).query_pairs(2.0,output_type='ndarray')
        if not len(pairs): break
        P = model.columns(xy,coeff)
        A = sparse.hstack((P,model.plane),format='csc')
        pars = np.array([fit_linear(A,im.ravel(),np.sqrt(c/im.ravel().clip(3)),len(xy))
                         for im,c in zip(images,counts)])
        predicted = (A@pars.T).T.reshape(images.shape)
        used=set()
        remove=[]
        for i,j in pairs[np.argsort(np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1))]:
            if i in used or j in used: continue
            center=np.average(xy[[i,j]],axis=0,weights=pars[:,[i,j]].mean(0).clip(.001))
            x,y=np.rint(center).astype(int)
            h,w=model.shape
            yy,xx=np.mgrid[max(0,y-5):min(h,y+6),max(0,x-5):min(w,x+6)]
            rows=(yy*w+xx).ravel()
            obs=images.reshape(nphase,-1)[:,rows]
            expected=predicted.reshape(nphase,-1)[:,rows]
            other=expected-(P[rows][:,[i,j]]@pars[:,[i,j]].T).T
            # Shot noise of the phase mean plus a 1.5% PSF interpolation/model allowance.
            noise=np.sqrt(expected.clip(3)/counts[:,None]+(.015*expected)**2)
            signal=(obs-other)/noise
            z=float(model.field_basis(center[None])[0]@coeff)
            def residual(v):
                shift=v[2:].reshape(nphase,2)
                pos=v[:2]+shift
                anchor=np.rint(pos).astype(int)
                off=pos-anchor
                kernels=model.kernels(pos,np.column_stack((np.full(nphase,np.clip(z,-.179,.179)),off[:,1],off[:,0])))
                dx=xx.ravel()-anchor[:,0,None]+model.radius
                dy=yy.ravel()-anchor[:,1,None]+model.radius
                side=2*model.radius+1
                inside=(dx>=0)&(dy>=0)&(dx<side)&(dy<side)
                basis=kernels[np.arange(nphase)[:,None],dy.clip(0,side-1),dx.clip(0,side-1)]*inside/noise
                amps=(basis*signal).sum(1)/(basis*basis).sum(1).clip(1e-12)
                # Small phase-specific centroid changes also occur for single finite beads.
                return np.r_[(signal-amps.clip(0)[:,None]*basis).ravel(),shift.ravel()/.04]
            single=optimize.least_squares(residual,np.r_[center,np.zeros(2*nphase)],
                                          max_nfev=30,diff_step=.0002)
            gain=2*single.cost-np.sum(((obs-expected)/noise)**2)
            if gain < (nphase+2)*np.log(obs.size):
                xy[i]=single.x[:2]
                remove.append(j)
                used.update((i,j))
        if not remove: break
        xy=np.delete(xy,remove,axis=0)
        xy,coeff,_,_=joint_positions(model,images,counts,xy,coeff,6)
    return xy,coeff


def movie_photometry(model,movie,xy,coeff):
    n = len(xy)
    base,dx,dy,_ = model.columns(xy,coeff,derivatives=True)
    A = sparse.hstack((base,model.plane),format='csc')
    fluxes,drift,background = [],[],[]
    for frame in movie:
        obs = frame.ravel().astype(float)
        # Expected signal provides the Poisson weights in the low-count pixels.
        first = fit_linear(A,obs,1/np.sqrt(np.maximum(obs,5)),n)
        w = 1/np.sqrt(np.maximum(A@first,3))
        motion = sparse.csc_matrix(np.stack((dx@first[:n],dy@first[:n]),axis=1))
        result = fit_linear(sparse.hstack((A,motion),format='csc'),obs,w,n)
        delta = result[-2:]
        shifted = sparse.hstack((model.columns(xy+delta,coeff),model.plane),format='csc')
        final = fit_linear(shifted,obs,w,n)
        fluxes.append(final[:n])
        background.append(final[n:])
        drift.append(delta)
    drift = np.array(drift)
    return np.array(fluxes),drift-drift.mean(0),np.array(background),drift.mean(0)


def axial_photometry(model, optics, movie, xy, coeff):
    """Estimate per-exposure focus offsets in um and lateral translations in pixels.

    Three vectorial PSF tables approximate the local focus dependence quadratically.
    Fit position and focus corrections jointly with the overlapping emitter amplitudes.
    """
    span = .045
    def at_focus(delta):
        settings = dict(optics, focus_um=optics['focus_um']+delta)
        return PixelModel(FiniteBeadRenderer(settings, model.radius), model.shape)
    lower, upper = at_focus(-span), at_focus(span)
    n = len(xy)
    reference, dx, dy, _ = model.columns(xy, coeff, derivatives=True)
    dz = (upper.columns(xy, coeff)-lower.columns(xy, coeff))/(2*span)
    A = sparse.hstack((reference, model.plane), format='csc')
    fluxes, backgrounds, lateral, axial = [], [], [], []
    for frame in movie:
        obs = frame.ravel().astype(float)
        first = fit_linear(A, obs, 1/np.sqrt(np.maximum(obs, 5)), n)
        weight = 1/np.sqrt(np.maximum(A@first, 3))
        motion = sparse.csc_matrix(np.stack([d@first[:n] for d in (dx, dy, dz)], axis=1))
        update = fit_linear(sparse.hstack((A, motion), format='csc'), obs, weight, n)
        shift = update[-3:-1]
        focus = np.clip(update[-1], -span, span)
        pos = xy+shift
        center = model.columns(pos, coeff)
        minus, plus = lower.columns(pos, coeff), upper.columns(pos, coeff)
        slope = (plus-minus)/(2*span)
        curve = (plus+minus-2*center)/(2*span**2)
        def design_at(z):
            return sparse.hstack((center+z*slope+z*z*curve, model.plane), format='csc')
        shifted = design_at(focus)
        flux = fit_linear(shifted, obs, weight, n)
        z_motion = sparse.csc_matrix(((slope+2*focus*curve)@flux[:n])[:, None])
        step = fit_linear(sparse.hstack((shifted, z_motion), format='csc'), obs, weight, n)
        focus = np.clip(focus+step[-1], -span, span)
        final = fit_linear(design_at(focus), obs, weight, n)
        fluxes.append(final[:n])
        backgrounds.append(final[n:])
        lateral.append(shift)
        axial.append(focus)
    lateral = np.asarray(lateral)
    return np.array(fluxes), lateral-lateral.mean(0), np.array(backgrounds), lateral.mean(0), np.array(axial)


class EmpiricalPSF(PixelModel):
    """Four normalized image-plane kernels, bilinearly mixed across the camera field."""
    def __init__(self, shape, radius):
        self.init_grid(shape,radius)
        g = np.exp(-(self.dx**2+self.dy**2)/(2*1.35**2))
        self.initial = g/g.sum()
        self.corners = np.repeat(self.initial[None],4,axis=0)
        self.refresh()

    def weights(self, xy):
        x,y = np.clip(xy/(np.array(self.shape[::-1])-1),0,1).T
        return np.column_stack(((1-x)*(1-y),x*(1-y),(1-x)*y,x*y))

    def refresh(self):
        shifts = np.linspace(-.53,.53,11)
        values = [[np.array([ndimage.shift(p,(dy,dx),order=3,mode='grid-constant').clip(0)
                             for p in self.corners]) for dx in shifts] for dy in shifts]
        values = np.array(values)
        values /= values.sum(axis=(-1,-2),keepdims=True)
        self.tables = RegularGridInterpolator((shifts,shifts),values)

    def kernels(self, xy, args):
        return np.einsum('bc,bcij->bij',self.weights(xy),self.tables(args[:,1:]))

    def update(self, xy, flux, residual, counts):
        # Regress each pixel residual on this bead's own phase brightness. Overlapping
        # neighbours and the planar background have already been subtracted by the joint fit.
        centers = xy.round().astype(int)
        keep = ((centers>=self.radius)&(centers<np.array(self.shape[::-1])-self.radius)).all(1)
        keep &= flux.mean(0)>np.median(flux.mean(0))*.20
        if keep.sum()<12:
            raise ValueError('At least 12 sufficiently bright whole bead footprints are needed to learn the PSF.')
        pos = xy[keep]
        centers = centers[keep]
        x = centers[:,0,None,None]+self.dx
        y = centers[:,1,None,None]+self.dy
        traces = flux[:,keep]
        weights = counts[:,None]*traces
        delta = np.einsum('fn,fnij->nij',weights,residual[:,y,x])
        delta /= np.maximum((weights*traces).sum(0),1)[:,None,None]
        aligned = np.array([ndimage.shift(p,-off[::-1],order=3,mode='grid-constant')
                            for p,off in zip(delta,pos-centers)])
        # Robustly pool aligned residuals; damp unresolved neighbours' leverage.
        W = self.weights(pos)
        Y = aligned.reshape(len(pos),-1)
        robust = np.ones(len(pos))
        for _ in range(3):
            coef = np.linalg.solve(W.T@(robust[:,None]*W)+np.eye(4)*.4, W.T@(robust[:,None]*Y))
            err = np.sqrt(np.mean((Y-W@coef)**2,axis=1))
            robust = np.minimum(1,np.maximum(np.median(err)*1.4,1e-5)/np.maximum(err,1e-10))
        self.corners = (self.corners+.9*coef.reshape(self.corners.shape)).clip(0)
        self.corners /= self.corners.sum(axis=(1,2),keepdims=True)
        self.refresh()



@dataclass
class DenseMovieFit:
    """Photon counts have shape (frames, beads); camera positions and drift are in pixels."""
    psf: PixelModel
    xy: np.ndarray
    photons: np.ndarray
    drift: np.ndarray
    background: np.ndarray
    field_coeff: np.ndarray
    axial_drift_um: np.ndarray | None = None


def fit_bead_movie(movie, groups=None, *, optics=None, radius=7):
    """Single fixed-focus movie in photoelectrons. Initial Gaussian -> empirical field PSF,
    or vectorial field fit with a finite-bead optical configuration. Group IDs identify
    repeated conditions for the position/PSF fit. Each frame gets independent bead fluxes.
    Background columns are intercept and slopes in normalized field coordinates.
    """
    if torch.is_tensor(movie):
        movie = movie.detach().cpu().numpy()
    movie = np.asarray(movie, dtype=np.float32)
    if movie.ndim != 3 or len(movie) == 0 or min(movie.shape[1:]) < 2*radius+1 or radius < 2:
        raise ValueError("expected a nonempty (frames, height, width) movie covering the PSF footprint")
    if not np.isfinite(movie).all():
        raise ValueError("movie pixels must be finite")
    if optics is None:
        model = EmpiricalPSF(movie.shape[1:], radius)
        xy, coeff = deblend(model, movie, groups, model.update)
    else:
        model = PixelModel(FiniteBeadRenderer(optics, radius), movie.shape[1:])
        xy, coeff = deblend(model, movie, groups)
    axial = None
    if optics is not None and optics.get('fit_axial_drift', False):
        photons, drift, bg, offset, axial = axial_photometry(model, optics, movie, xy, coeff)
    else:
        photons, drift, bg, offset = movie_photometry(model, movie, xy, coeff)
    return DenseMovieFit(model, xy+offset, photons, drift, bg, coeff, axial)
