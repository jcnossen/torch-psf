"""Local efficiency check for the actual two-stage 96-spot Monte Carlo design."""
import copy
import csv
import numpy as np
import torch
from simulate import HERE, SEEDS, datasets, geometry, nat_matrix, tensor


@torch.no_grad()
def blocks(psf, xyz, field):
    xyz=xyz.double()
    field=field.double()
    def signal(pos, coef):
        return psf(pos,.15,zcoef_offset=coef).flatten(1)
    raw=signal(xyz,field)
    inv_sd=(3000*raw+8).rsqrt()
    local=[]
    for delta in torch.diag(xyz.new_tensor([.001,.001,.0001])):
        local.append(1500*(signal(xyz+delta,field)-signal(xyz-delta,field))/delta.max())
    local.extend([raw,torch.ones_like(raw)])
    optical=[]
    for delta in torch.eye(6,device=xyz.device,dtype=xyz.dtype)*.00001:
        optical.append(1500*(signal(xyz,field+delta)-signal(xyz,field-delta))/delta.max())
    L=torch.stack(local,-1)*inv_sd[:,:,None]
    A=torch.stack(optical,-1)*inv_sd[:,:,None]
    intrinsic=torch.linalg.inv(L.mT@L)
    response=intrinsic@L.mT@A
    residual=A-L@response
    return tuple(x.cpu().numpy() for x in (intrinsic[:,:3,:3],response[:,:3],residual.mT@residual))


def main():
    covs={}
    for seed,scenario,psf,_,true,D,_,_,xyz,E,_ in datasets(SEEDS):
        model=copy.deepcopy(psf).double()
        _,_,cal_xyz=geometry(np.random.default_rng(seed),len(D))
        _,_,S=blocks(model,cal_xyz,D.double()@tensor(true).double())
        V,R,_=blocks(model,xyz,E.double()@tensor(true).double())
        for name,B in [('Known pupil',None),('NAT',nat_matrix()),('Independent',np.eye(25))]:
            if name=='NAT' and scenario!='NAT field':
                continue  # Truth lies outside the model; ordinary unbiased CRLB is inapplicable.
            cov=V.copy()
            if B is not None:
                G=D.cpu().numpy()@B
                info=(G.transpose(0,2,1)@S@G).sum(0)
                H=R@E.cpu().numpy()@B
                cov+=H@np.linalg.inv(info)@H.transpose(0,2,1)
            covs.setdefault((scenario,name),[]).append(cov.diagonal(axis1=1,axis2=2)*[108**2,108**2,1000**2])
    with (HERE/'matched_precision.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['truth','model','xy_local_sd_nm','z_local_sd_nm'])
        for key,v in covs.items():
            var=np.mean(v,axis=(0,1))
            row=[*key,np.sqrt(var[:2].mean()),np.sqrt(var[2])]
            print(row)
            writer.writerow(row)


if __name__=='__main__':
    main()
