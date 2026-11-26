"""Export the native Vectorfit 0.3.4 field routine for a cross-runtime parity check."""
import csv
from pathlib import Path
import numpy as np
import vectorfit_py as vf

def main():
    if vf.__version__ != '0.3.4':
        raise ValueError(f'Expected vectorfit-py 0.3.4, got {vf.__version__}')
    modes = vf.aberrationArray1D(6)
    for j,(n,m) in enumerate([(2,0),(2,-2),(2,2),(3,-1),(3,1),(4,0)]):
        mode = vf.Aberration()
        mode.radial_order,mode.azimuthal_order,mode.coefficient = n,m,0.
        modes.set(mode,(j,))
    xy = np.array([(x,y) for x in (-1.,-.4,0.,.7,1.) for y in (-1.,-.3,0.,.6,1.)])
    with Path(__file__).with_name('vectorfit_field_reference.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['gamma','u','v','Z4','Z5','Z6','Z7','Z8','Z11'])
        for k,g in enumerate(np.eye(13),1):
            vals=vf.compute_xy_gamma_legendre(*[vf.doubleArray1D(x.copy()) for x in (xy[:,0],xy[:,1],g)],modes)
            for p,row in zip(xy,np.asarray(vals)):
                writer.writerow([k,*p,*row])

if __name__=='__main__':
    main()
