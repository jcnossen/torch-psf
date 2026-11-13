import time, torch, sys
sys.path.insert(0, 'study/insitu')
import common as C
from torch_convspline.knotpsf import KnotPSF, localize, crlb
t=time.time()
movie, truth = C.sim_movie('A', T=2000)
print('movie', movie.shape, 'events', len(truth), 'per frame', len(truth)/2000, f'{time.time()-t:.1f}s')
t=time.time()
rois, fr, cor = C.detect_crop(movie)
print('detected', len(rois), f'{time.time()-t:.1f}s')
# truth spline
spl = KnotPSF(C.P, *C.ZR, S=2, hz=0.04, device=C.DEV)
spl.knots = spl.stack_from_vectorial(C.make_vec(spl.Pl, 'A', 'insitu'), C.insitu_pos)
t=time.time()
th, nll = localize(spl, rois)
torch.cuda.synchronize(); print('localize', f'{time.time()-t:.1f}s')
# match to truth: same frame, nearest
X = cor[:,0] + (C.P-1)//2 + th[:,0]; Y = cor[:,1] + (C.P-1)//2 + th[:,1]
key_t = truth[:,0].long()
# brute force by frame via sorting
import numpy as np
tr = truth.cpu().numpy(); est = torch.stack([fr.float(), X, Y, th[:,2], th[:,3]],1).cpu().numpy()
from collections import defaultdict
byf = defaultdict(list)
for i,r in enumerate(tr): byf[int(r[0])].append(i)
err=[]; nclose=[]
for e in est:
    c = byf.get(int(e[0]), [])
    if not c: continue
    d = np.hypot(tr[c,1]-e[1], tr[c,2]-e[2]); j = c[int(d.argmin())]
    nclose.append((d < 10).sum())
    if d.min() < 2: err.append([e[1]-tr[j,1], e[2]-tr[j,2], e[3]-tr[j,3], e[4]/tr[j,4]])
err=np.array(err); nclose=np.array(nclose)
print('matched', len(err), 'of', len(est), ' rois with >1 emitter within 10px:', (nclose>1).mean())
print('rmse x,y (nm)', err[:,:2].std(0)*108, 'z (nm)', err[:,2].std()*1000, 'bias', err[:,:3].mean(0)*[108,108,1000], 'I ratio', np.median(err[:,3]))
cr = crlb(spl, th).cpu().numpy()
print('median crlb x,y nm', np.median(cr[:,:2],0)*108, 'z nm', np.median(cr[:,2])*1000)
# test set
ts = C.sim_test('A', 2000)
th2, _ = localize(spl, ts['d'])
e = (th2 - ts['th'])
print('test rmse nm', e[:,0].std().item()*108, e[:,1].std().item()*108, e[:,2].std().item()*1000, 'robust z', (e[:,2].abs().median()*1483).item())
