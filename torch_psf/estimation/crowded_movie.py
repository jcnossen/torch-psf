"""GPU weighted image fits for crowded, single-layer bead movies with shared optics."""
import numpy as np
import torch
from scipy import ndimage, spatial
from .bead_movie import DenseMovieFit, grouped_images, propose
from .dense_beads import find_all_beads
from .vectorial_movie_table import VectorialMovieTable


class Scene:
    def __init__(self, cfg, shape, radius):
        self.table = VectorialMovieTable(cfg, radius)
        self.tensor = self.table.tensor
        self.h,self.w = shape
        self.size = self.h*self.w
        self.span = self.tensor([self.w-1,self.h-1])
        dy,dx = np.mgrid[-radius:radius+1,-radius:radius+1]
        self.dx,self.dy = self.tensor(dx.ravel()).long(),self.tensor(dy.ravel()).long()
        yy,xx = torch.meshgrid(self.tensor(np.linspace(-1,1,self.h)),self.tensor(np.linspace(-1,1,self.w)),indexing="ij")
        self.bg_basis = torch.stack([torch.ones_like(xx),xx,yy]).reshape(3,-1)

    def geometry(self, pos, coeff, focus=0., derivatives=False):
        anchor=pos.round().long()
        x=anchor[:,0,None]+self.dx
        y=anchor[:,1,None]+self.dy
        valid=(x>=0)&(x<self.w)&(y>=0)&(y<self.h)
        rows=(y*self.w+x).clamp(0,self.size-1)
        basis=torch.cat([torch.ones_like(pos[:,:1]),2*pos/self.span-1],1)
        wave=(basis@coeff).clamp(-.175,.175)
        offset=pos-anchor
        def kernel(off,w,z):
            return self.table.kernels(off,w,z)*valid
        base=kernel(offset,wave,focus)
        if not derivatives:
            return rows,base
        deriv=[]
        for direction in ((.008,0),(0,.008)):
            delta=self.tensor(direction)
            deriv.append((kernel(offset+delta,wave,focus)-kernel(offset-delta,wave,focus))/.016)
        deriv.append((kernel(offset,wave+.001,focus)-kernel(offset,wave-.001,focus))/.002)
        zlo,zhi=max(self.table.focus[0], focus-.002), min(self.table.focus[-1], focus+.002)
        deriv.append((kernel(offset,wave,zhi)-kernel(offset,wave,zlo))/(zhi-zlo))
        return rows,base,deriv,basis

    def project(self, rows, patches):
        out=patches.new_zeros((len(patches),self.size))
        return out.scatter_add_(1,rows.flatten().expand(len(out),-1),patches.flatten(1))

    def solve_flux(self,obs,rows,psf,flux,bg,iterations):
        for i in range(iterations+1):
            pred=(self.project(rows,psf*flux[:,:,None])+bg@self.bg_basis).clamp_min(2)
            resid=(obs-pred)/pred
            if i==iterations: break
            # Per-pixel Poisson allocation gives a positive simultaneous update for all overlapping beads.
            mult=(resid[:,rows]*psf).sum(-1)/psf.sum(-1).clamp_min(.01)
            flux=(flux*(1+mult)).clamp_min(.5)
            weight=pred.reciprocal()
            gram=(self.bg_basis[None]*weight[:,None])@self.bg_basis.T
            delta=torch.linalg.solve(gram,(resid@self.bg_basis.T).unsqueeze(-1)).squeeze(-1)
            bg=bg+.35*delta
            bg[:,0].clamp_(min=1)
        return flux,bg,pred,resid

    def groups_fit(self,images,xy,coeff,flux,bg,steps):
        obs=self.tensor(images.reshape(len(images),-1))
        for step in range(steps):
            rows,psf,d,basis=self.geometry(xy,coeff,derivatives=True)
            flux,bg,pred,resid=self.solve_flux(obs,rows,psf,flux,bg,3)
            grad=(resid[:,rows]*flux[:,:,None]).sum(0)
            weight=(flux.square()[:,:,None]/pred[:,rows]).sum(0)
            # Profile a common amplitude direction out of the local 2D position update.
            norm=(weight*psf.square()).sum(1).clamp_min(1e-8)
            dx,dy=[g-psf*((weight*psf*g).sum(1)/norm)[:,None] for g in d[:2]]
            a=(weight*dx.square()).sum(1)+1
            b=(weight*dy.square()).sum(1)+1
            c=(weight*dx*dy).sum(1)
            gx,gy=(grad*dx).sum(1),(grad*dy).sum(1)
            det=(a*b-c.square()).clamp_min(1e-4)
            move=torch.stack([(b*gx-c*gy)/det,(a*gy-c*gx)/det],1)
            xy=(xy+.65*move.clamp(-.45,.45)).clamp_min(0).minimum(self.span)
            field=torch.stack([self.project(rows,flux[:,:,None]*d[2]*v[None,:,None]) for v in basis.T],1)
            gram=(field/pred[:,None])@field.transpose(1,2)
            rhs=(field*resid[:,None]).sum((0,2))
            delta=torch.linalg.solve(gram.sum(0)+torch.eye(3,device=xy.device),rhs)
            coeff=(coeff+.55*delta.clamp(-.008,.008)).clamp(-.15,.15)
        rows,psf=self.geometry(xy,coeff)
        flux,bg,pred,_=self.solve_flux(obs,rows,psf,flux,bg,4)
        return xy,coeff,flux,bg,pred

    def exposure(self,frame,xy,coeff,flux,bg,reference):
        obs=self.tensor(frame.reshape(1,-1))
        motion=self.tensor([0,0,0])
        for iteration in range(3):
            focus=float(motion[2])
            rows,psf,d,_=reference if iteration==0 else self.geometry(xy+motion[:2],coeff,focus,True)
            flux,bg,pred,resid=self.solve_flux(obs,rows,psf,flux,bg,5)
            deriv=torch.stack([self.project(rows,flux[:,:,None]*g)[0] for g in (d[0],d[1],d[3])])
            step=torch.linalg.solve((deriv/pred)@deriv.T+torch.eye(3,device=xy.device),deriv@resid[0])
            motion+=step
            motion[2].clamp_(self.table.focus[0], self.table.focus[-1])
        rows,psf=self.geometry(xy+motion[:2],coeff,float(motion[2]))
        flux,bg,*_=self.solve_flux(obs,rows,psf,flux,bg,7)
        return flux[0],bg[0],motion


@torch.no_grad()
def fit_crowded_movie(movie,groups,optics,radius=7):
    """Fit vectorial Z6, bead positions and photon counts, planar backgrounds, and XYZ drift."""
    images,counts=grouped_images(movie.astype(np.float32),groups)
    inv = np.arange(len(movie)) if groups is None else np.unique(groups, return_inverse=True)[1]
    scene=Scene(optics,movie.shape[1:],radius)
    peaks,_=find_all_beads(images,sigma=.6,threshold=.9,min_dist=1,bg_size=9)
    if not len(peaks): raise ValueError("No bead candidates")
    xy=scene.tensor(peaks)
    coeff=scene.tensor(np.zeros(3))
    flux=scene.tensor(np.full((len(images),len(xy)),5500))
    bg=scene.tensor(np.tile([18,0,0],(len(images),1)))
    for stage,steps in enumerate((20,16,16,16,20)):
        print("Group fit",stage+1,"candidates",len(xy),flush=True)
        xy,coeff,flux,bg,pred=scene.groups_fit(images,xy,coeff,flux,bg,steps)
        if stage==4: break
        expected=pred.cpu().numpy().reshape(images.shape)
        residual=images-expected
        score=(ndimage.gaussian_filter(residual,(0,.7,.7))/np.sqrt(expected.clip(3)/counts[:,None,None])).max(0)
        add=propose(score,6.5,xy.cpu().numpy())
        if not len(add): break
        xy=torch.cat((xy,scene.tensor(add)))
        flux=torch.cat((flux,scene.tensor(np.full((len(images),len(add)),550))),1)
    keep=flux.mean(0)>max(100,.07*float(flux.mean(0).median()))
    xy,flux=xy[keep],flux[:,keep]
    xy,coeff,flux,bg,_=scene.groups_fit(images,xy,coeff,flux,bg,10)
    reference=scene.geometry(xy,coeff,derivatives=True)
    phot,background,motions=[],[],[]
    for t,frame in enumerate(movie):
        if t%25==0: print("Exposure",t+1,flush=True)
        f,b,m=scene.exposure(frame,xy,coeff,flux[inv[t]:inv[t]+1],bg[inv[t]:inv[t]+1],reference)
        phot.append(f.cpu().numpy()); background.append(b.cpu().numpy()); motions.append(m.cpu().numpy())
    motion=np.array(motions)
    mean=motion[:,:2].mean(0)
    return DenseMovieFit(scene.table,xy.cpu().numpy()+mean,np.array(phot),motion[:,:2]-mean,
                         np.array(background),coeff.cpu().numpy(),motion[:,2])
