"""Compare first-order NAT and independent field surfaces using pixel-level local information.

Requires numpy, matplotlib, torch and torch_psf (tested with 8b70122).
Run python compare.py to regenerate the adjacent PDF. All distances below are in um.
"""
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch_psf.field import FieldZernikes
from torch_psf.vectorial import VectorialPSF

MODES = [4, 5, 6, 7, 8, 11]
ORDERS = [2, 2, 2, 1, 1, 0]
SLOTS = [(j, p) for j, order in zip(MODES, ORDERS)
         for p in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)] if sum(p) <= order]
# Each entry is (Noll mode, u power, v power, multiplier). Input is gamma-bar
# in waves for the literal unnormalized pupil polynomials of S42-43.
s, d, a = np.sqrt(3), np.sqrt(5), 1.5 * np.sqrt(15 / 7)
TERMS = [
    [(4,0,0,1)], [(4,1,0,s)], [(4,0,1,s)],
    [(4,0,0,-d),(4,2,0,1.5*d),(4,0,2,1.5*d)],
    [(6,0,0,1)], [(5,0,0,1)],
    [(6,1,0,s),(5,0,1,s)], [(6,0,1,-s),(5,1,0,s)],
    [(6,2,0,a),(6,0,2,-a),(5,1,1,2*a)],
    [(8,0,0,1)], [(7,0,0,1)], [(8,1,0,s),(7,0,1,s)], [(11,0,0,1)]]
PUPIL_NORM = dict(zip(MODES, np.sqrt([3,6,6,8,8,5])))


def nat_matrix():
    matrix = np.zeros((len(SLOTS), 13))
    for k, terms in enumerate(TERMS):
        for j, i, h, value in terms:
            matrix[SLOTS.index((j,(i,h))), k] = value / PUPIL_NORM[j]
    assert np.linalg.matrix_rank(matrix) == 13
    return matrix


def field_design(u,v):
    return np.array([[u**i * v**h if j == mode else 0. for j,(i,h) in SLOTS] for mode in MODES])


def check_library():
    # Exercise the real parameter ordering and reference subtraction on all 13 columns.
    xy = np.array([[-1.,1.],[.4,-.7],[0.,0.],[1.,1.]])
    field = FieldZernikes(MODES, dict(zip(MODES,ORDERS)))
    for col in nat_matrix().T:
        coef = dict(zip(SLOTS,col))
        with torch.no_grad():
            field.coef.copy_(torch.tensor([[coef.get((j,tuple(p)),0.) for p in field.powers[1:].int().tolist()] for j in MODES]))
        base = np.array([coef[j,(0,0)] for j in MODES])
        actual = base + field.relative(xy).detach().numpy()
        np.testing.assert_allclose(actual,[field_design(*p)@col for p in xy],atol=5e-7)


def pixel_information():
    psf = VectorialPSF(15, pixel_um=.108, NA=1.4, n_imm=1.518, n_med=1.33,
                       wavelength_um=.68, zernike_indices=MODES, npupil=48,
                       oversample=3, device='cpu', pixel_integration='exact')
    # Derivatives in nm for xyz, waves for pupil parameters.
    scale = np.array([1/108,1/108,.001])
    def image(z, delta):
        pos = torch.tensor([.17,-.23,z] + scale * delta[:3], dtype=torch.float32)[None]
        coef = torch.tensor(np.array([0,0,.08,0,0,0])+delta[5:],dtype=torch.float32)[None]
        signal = psf(pos,.15,zcoef_offset=coef).flatten().detach().numpy().astype(float)
        return (3000+delta[3])*signal + 8 + delta[4]
    output = []
    for z in [.05,.15,.25]:
        zero = np.zeros(11)
        mu = image(z,zero)
        def derivative(step):
            return (image(z,step)-image(z,-step))/(2*max(step))
        jac = np.stack([derivative(step) for step in np.diag([.4,.4,.4,1,.1]+[.0003]*6)],axis=1)
        white = jac / np.sqrt(mu[:,None])
        local, optics = white[:,:5], white[:,5:]
        response = np.linalg.lstsq(local,optics,rcond=None)[0]
        residual = optics-local@response
        output.append((residual.T@residual,response[:3],np.linalg.inv(local.T@local)[:3,:3]))
    return output


def compare():
    torch.set_num_threads(1)
    check_library()
    info = pixel_information()
    field_points = [field_design(u,v) for u in np.linspace(-1,1,7) for v in np.linspace(-1,1,7)]
    # Four independent calibration emitters per field point and depth.
    fisher = sum(4*F.T@I@F for F in field_points for I,_,_ in info)
    violation = np.zeros(25)
    violation[SLOTS.index((8,(0,1)))] = .010  # 10 milliwaves of cross-axis coma at v=1
    tests = [field_design(u,v) for u in np.linspace(-.95,.95,16) for v in np.linspace(-.95,.95,16)]
    rows = []
    for name,B,ridge in [('NAT',nat_matrix(),False),('Independent',np.eye(25),False),
                         ('Independent + ridge',np.eye(25),True)]:
        penalty = np.diag([400. if ridge and sum(p)>0 else 0. for _,p in SLOTS])
        gain = B @ np.linalg.solve(B.T@(fisher+penalty)@B,B.T)
        covariance = gain@fisher@gain.T
        error = (gain@fisher-np.eye(25))@violation
        for metric in ['field','xy','z']:
            var, bias2 = [], []
            for F in tests:
                for _,R,V in info:
                    if metric=='field':
                        J, intrinsic = 1000*F, np.zeros(6)
                    else:
                        axes = [0,1] if metric=='xy' else [2]
                        J, intrinsic = R[axes]@F, V.diagonal()[axes]
                    var.extend(np.diag(J@covariance@J.T)+intrinsic)
                    bias2.extend((J@error)**2)
            rows.append((name,metric,np.sqrt(np.mean(var)),np.sqrt(np.mean(bias2))))
    return rows


def report(rows):
    import textwrap
    from math import hypot

    target = Path(__file__).with_name('droste_field_comparison.pdf')
    ink, teal = '#193342', '#00696b'
    def page(title, subtitle, number):
        fig = plt.figure(figsize=(8.27,11.69))
        fig.text(.08,.95,'TorchPSF   /   METHODS',color=teal,weight='bold',size=10)
        fig.text(.08,.903,title,size=22,color=ink,weight='bold')
        fig.text(.08,.872,subtitle,size=10,color=ink)
        fig.text(.08,.039,'Droste et al. and torch_psf   /   Method comparison',size=9,color=ink)
        fig.text(.9,.039,str(number),size=9,color=ink)
        return fig
    def body(fig,y,txt):
        fig.text(.08,y,textwrap.fill(txt,95),va='top',fontsize=10,linespacing=1.38,color=ink)
    def table(fig,bounds,headers,data,widths):
        ax=fig.add_axes(bounds)
        ax.axis('off')
        t=ax.table(cellText=data,colLabels=headers,colWidths=widths,cellLoc='left',colLoc='left',bbox=[0,0,1,1])
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        for (r,c),cell in t.get_celld().items():
            cell.set_linewidth(.5)
            cell.set_edgecolor('white')
            cell.set_facecolor(teal if r==0 else ('#e8f1f0' if r%2 else '#f4f6f5'))
            cell.set_text_props(color='white' if r==0 else ink)
    with PdfPages(target) as pdf:
        f=page('Thirteen parameters, six pupil modes','First-order nodal aberration theory expressed in torch_psf coordinates',1)
        body(f,.831,'Each row adds one fitted parameter g to the listed slots. Here g is the corresponding barred gamma in Supplement 1, Eq. S58, expressed in waves. In the table, s = √3, d = √5 and a = (3/2)√(15/7). The entries are numerators. Divide each contribution by its mode factor below when mapping the literal pupil polynomials of Eq. S43.')
        entries=[
            ('1','Defocus','4[1] += g'),('2','Defocus','4[u] += sg'),('3','Defocus','4[v] += sg'),
            ('4','Field curvature','4[1] += −dg,  4[u²], 4[v²] += 1.5dg'),
            ('5','Axial astigmatism','6[1] += g'),('6','Diagonal astigmatism','5[1] += g'),
            ('7','Astigmatism slopes','6[u], 5[v] += sg'),('8','Astigmatism slopes','6[v] += −sg,  5[u] += sg'),
            ('9','Astigmatism curvature','6[u²] += ag,  6[v²] += −ag,  5[uv] += 2ag'),
            ('10','Horizontal coma','8[1] += g'),('11','Vertical coma','7[1] += g'),
            ('12','Radial coma slope','8[u], 7[v] += sg'),('13','Primary spherical','11[1] += g')]
        table(f,[.08,.335,.84,.39],['g index','Aberration','Noll mode [monomial]'],entries,[.13,.30,.57])
        body(f,.30,'Pupil scaling. For modes 4, (5, 6), (7, 8), and 11 divide by √3, √6, √8, and √5 respectively. Eq. S43 prints unit-edge radial polynomials, whereas torch_psf uses unit-RMS pupil modes. Eq. S32 also describes the paper’s coefficients as RMS. For exported coefficients already in RMS waves, omit the division. Confirm that convention and pupil handedness before importing numerical fits.')
        body(f,.188,'Field coordinates u and v run from −1 to +1, aligned with camera columns and rows and the corresponding pupil axes. The default reference is the field centre. A slot [1] belongs to base.zcoef. The order-two field.coef columns are [v, v², u, uv, u²]. All unlisted contributions are zero. For a different reference, put the evaluated surface there into base.zcoef and retain the polynomial slopes.')
        pdf.savefig(f); plt.close(f)

        f=page('Which coefficients can move together?','First-order NAT is a rank-13 subspace of a degree-matched 25-parameter field.',2)
        body(f,.832,'The independent model fits six coefficients each for defocus and the two astigmatisms, three each for the two comas, and one for spherical aberration. This frees 25 parameters in torch_psf. The comparison gives both models identical per-mode maximum degree, optics, and camera coordinates.')
        table(f,[.08,.51,.84,.215],['Mode family','Constraints in aligned, normalized coordinates'],[
            ('Defocus','4[u²] = 4[v²],  4[uv] = 0'),
            ('Astigmatism, linear','6[u] = 5[v],  6[v] = −5[u]'),
            ('Astigmatism, quadratic','6[u²] = −6[v²] = 5[uv]/2'),
            ('Astigmatism, zeros','6[uv] = 5[u²] = 5[v²] = 0'),
            ('Coma','8[u] = 7[v],  8[v] = 7[u] = 0'),
            ('Spherical','Constant in both degree-matched models')],[.31,.69])
        body(f,.475,'The constraints remove six slots (one defocus, three astigmatism, two coma) and six independent equalities (one defocus, four astigmatism, one coma). The resulting dimension is 25 − 6 − 6 = 13.')
        body(f,.37,'The defocus intercept includes −√5 g4 because P₂(u) + P₂(v) = 1.5(u² + v²) − 1. For astigmatism, P₂(u) − P₂(v) cancels the constants. Eq. S57 prints a positive equality between the two quadratic astigmatism terms. Eqs. S50, S53 and the final Eq. S58 agree on the opposite signs used here.')
        body(f,.254,'Implementation check. The script constructs the 25 × 13 map, verifies rank 13 and evaluates every column against FieldZernikes.relative plus the base coefficient at four field positions. For an actual fit, train the 13 parameters and expand them on each forward pass. Initializing independent slots with equal values only enforces the ties at initialization.')
        f.text(.08,.128,'Sources',color=teal,weight='bold')
        f.text(.08,.10,'Droste et al., Optica 12, 1220–1229 (2025)  •  Article',url='https://doi.org/10.1364/OPTICA.560721',size=10,color=teal)
        f.text(.08,.075,'Supplement 1, §§2.1–2.2, Eqs. S30–S58  •  Published supplement',url='https://doi.org/10.6084/m9.figshare.29237045.v2',size=10,color=teal)
        pdf.savefig(f); plt.close(f)

        f=page('Precision and bias under matched optics','Pixel-level Fisher analysis  /  independent calibration and test emitters',3)
        body(f,.831,'All three fits see the same Poisson pixels. The reference pupil has 0.08 waves RMS of Noll 6, with all other coefficients zero. At this NAT-compatible field the local bias is zero for all three fits. To expose model bias, add 0.010 waves × v to horizontal coma (Noll 8). This violates one NAT zero. The table gives first-order predictions around the reference pupil.')
        data=[[name,{'field':'Field (mλ)','xy':'xy (nm)','z':'z (nm)'}[metric],f'{sd:.3f}',f'{bias:.3f}',f'{hypot(sd,bias):.3f}'] for name,metric,sd,bias in rows]
        table(f,[.08,.48,.84,.23],['Field model','Quantity','RMS SD','RMS bias¹','RMSE¹'],data,[.30,.23,.15,.16,.16])
        body(f,.453,'¹ Bias and RMSE use the cross-coma perturbation. RMS SD applies to both truths in this local approximation. Field errors average over six Noll surfaces in milliwaves. Position errors average over test sites and depths, with x and y pooled per coordinate. Position SD includes photon noise of a new emitter and uncertainty in the fitted field.')
        body(f,.341,'Design. 588 calibration emitters on a uniform 7 × 7 field grid, four per site at each height (50, 150, 250 nm). Test at 16 × 16 interior sites and the same heights. Each spot has 3,000 nominal photons and 8 background photons/pixel. The 15 × 15 ROI uses 108 nm pixels, NA 1.4, 680 nm emission, medium/immersion indices 1.33/1.518, focus 150 nm, and fixed lateral offsets (+0.17, −0.23) pixels.')
        body(f,.212,'At each depth we differentiate the torch_psf vectorial renderer and weight by inverse Poisson variance. We project out each calibration emitter’s x, y, z, intensity and background before pooling field information. The ridge fit penalizes nonconstant monomials with prior SD 0.05 waves. Covariance uses the sampling variance of the penalized fit. No field-selection step is applied.')
        body(f,.097,'These are local precision bounds and linearized biases. Finite-sample optimizer failures, axial sign ambiguity, spline tiling and experimental ground truth require separate measurements.')
        pdf.savefig(f); plt.close(f)

        f=page('Where the trade-off changes','Effect of a cross-axis coma slope on lateral and axial localization',4)
        amplitudes=np.linspace(0,20,101)
        for index,metric in enumerate(['xy','z']):
            ax=f.add_axes([.13,.60-.31*index,.77,.19])
            for name,m,sd,bias in rows:
                if m==metric:
                    ax.plot(amplitudes,np.hypot(sd,bias*amplitudes/10),label=name)
            ax.set(xlabel='Horizontal coma change at v = 1 (mλ RMS)',ylabel=metric+' RMSE (nm)')
            ax.grid(alpha=.22)
            if index==0:
                ax.legend(frameon=False,fontsize=8)
        body(f,.20,'Imposing the full NAT constraint set improves axial precision in this design. A vertical gradient of horizontal coma falls outside that set and produces a lateral shift. At 10 mλ the independent fit gives lower lateral RMSE, while NAT retains lower axial RMSE. The appropriate model therefore depends on which localization error matters and which field terms the instrument exhibits.')
        body(f,.095,'Curves extrapolate the first-order response with fixed Fisher information. They illustrate the direction and scale of the trade-off near this pupil. Other aberrations, field coverage and photon budgets change the crossing points.')
        pdf.savefig(f); plt.close(f)


        # Empirical results are produced by simulate.py and shipped beside this report.
        import csv
        def read_results(name):
            with Path(__file__).with_name(name).open(newline='') as stream:
                return list(csv.DictReader(stream))
        summary = read_results('simulation_summary.csv')
        runs = read_results('simulation_runs.csv')
        f=page('Localization on simulated images','Repeated calibration and localization with known emitter positions',5)
        body(f,.83,'We generated 12 independent calibration movies per truth, represented as isolated spot ROIs. Each contains 96 single-emitter images. Each fitted field was frozen and used to localize 128 newly exposed test spots. Both models and the ridge variant received identical counts within each trial. Test positions were held fixed across trials so repeatability can be measured at each emitter.')
        entries=[]
        for row in summary:
            entries.append([row['scenario'].replace(' field',''),row['model']]+
                           [f'{float(row[key]):.2f}' for key in ['field_rmse','xy_sd','z_sd','xy_rmse','z_rmse']])
        table(f,[.07,.535,.86,.175],['Truth','Fit','Field¹','xy SD','z SD','xy RMSE','z RMSE'],
              entries,[.145,.235,.12,.12,.12,.13,.13])
        body(f,.506,'¹ Field RMSE is in milliwaves, pooled over 6 modes and 128 test sites. Position errors are in nm. SD is the root mean of the within-site sample variances over 12 independent calibrations and exposures. RMSE uses every error against ground truth. Lateral metrics pool x and y per coordinate.')
        body(f,.393,'Simulation design. Both truths include nonzero NAT slopes and curvature on a nominal 0.08-wave astigmatic pupil. The second adds 0.030 waves × v to Noll 8. Each spot has 3,000 nominal photons, 8 background photons/pixel and independent Poisson counts. Heights are uniform over 55–245 nm, field coordinates uniform over the square and lateral offsets within ±0.35 pixel. Optical constants match page 3. We use a 32-sample pupil and 2× image oversampling for generation and fitting.')
        body(f,.231,'Calibration jointly fits each emitter’s x, y, z, signal and background with either 13 or 25 shared field parameters by Poisson likelihood. All fits start from the nominal astigmatic pupil, centered spots, z = 150 nm, 3,000 photons and background 8. Test fits estimate only the five spot parameters. The ridge variant uses a 0.05-wave prior SD on nonconstant field terms.')
        body(f,.119,'This experiment isolates the field constraint with a shared nonlinear optimizer and matched renderer. It assumes known ROI segmentation and field coordinates. The optimization implementations in Vectorfit and the tiled torch_psf estimator can introduce additional differences.')
        pdf.savefig(f); plt.close(f)

        f=page('Field-dependent localization bias','Signed errors and paired uncertainty across independent calibrations',6)
        entries=[]
        for row in summary:
            entries.append([row['scenario'].replace(' field',''),row['model']]+
                           [f'{float(row[key]):+.2f}' for key in ['cross_coma','x_slope','z_bias']])
        table(f,[.07,.637,.86,.175],['Truth','Fit','Coma slope¹','x bias slope²','Mean z error³'],
              entries,[.16,.25,.19,.20,.20])
        body(f,.605,'¹ Fitted Noll 8 coefficient of v, in milliwaves per normalized field unit. True slopes are 0 and +30. ² Regression slope of signed x localization error against v (nm/unit), averaged over trials. Opposite errors at opposite edges can cancel in the mean. ³ Signed axial error averaged across test emitters and trials (nm).')
        rng=np.random.default_rng(12019)
        entries=[]
        for scenario in dict.fromkeys(r['scenario'] for r in runs):
            def series(model,key):
                paired=sorted((r for r in runs if r['scenario']==scenario and r['model']==model),
                              key=lambda r:int(r['seed']))
                return np.array([float(r[key]) for r in paired])
            for key,label in [('xy_rmse','Lateral RMSE'),('z_rmse','Axial RMSE'),('x_slope','x bias slope')]:
                delta=series('NAT',key)-series('Independent',key)
                boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
                lo,hi=np.percentile(boot,[2.5,97.5])
                entries.append([scenario.replace(' field',''),label,f'{delta.mean():+.2f}',f'{lo:+.2f} to {hi:+.2f}'])
        table(f,[.08,.327,.84,.173],['Truth','NAT minus independent','Mean Δ','95% interval'],entries,[.20,.35,.15,.30])
        body(f,.291,'Intervals resample the 12 paired dataset differences 10,000 times. The estimand is the mean of the per-dataset differences, conditional on this test layout. Positive RMSE differences favor the independent fit. A signed slope describes the direction of a field trend.')
        statuses=[int(r[k]) for r in runs for k in ('cal_status','test_status')]
        body(f,.175,f'Optimizer diagnostics. {statuses.count(0)} of {len(statuses)} fits passed the double-precision refinement. Per-dataset results include projected gradients, likelihood improvements and both optimizer stages. For reproduction run simulate.py, audit.py and compare.py with torch_psf on the Python path.')
        pdf.savefig(f); plt.close(f)
        f=page('Variation across datasets','Paired measurements from each of the 12 calibration and test realizations',7)
        scenarios=list(dict.fromkeys(r['scenario'] for r in runs))
        for i,(key,label) in enumerate([('xy_rmse','Lateral RMSE (nm)'),('z_rmse','Axial RMSE (nm)')]):
            for j,scenario in enumerate(scenarios):
                ax=f.add_axes([.11+.47*j,.58-.32*i,.34,.20])
                grouped={}
                for r in runs:
                    if r['scenario']==scenario:
                        grouped.setdefault(r['seed'],{})[r['model']]=float(r[key])
                paired=np.array([[pair[m] for m in ('NAT','Independent')] for pair in grouped.values()])
                ax.plot([0,1],paired.T,color='#78979a',alpha=.63,linewidth=.85,marker='o',ms=2.6)
                ax.plot([0,1],paired.mean(0),color=teal,linewidth=2.4,marker='o',ms=4)
                ax.set(xlim=(-.2,1.2),xticks=[0,1],xticklabels=['NAT','Independent'],ylabel=label)
                if i==0:
                    ax.set_title(scenario,fontsize=11)
                ax.tick_params(labelsize=8)
                ax.grid(axis='y',alpha=.18)
                ax.spines[['top','right']].set_visible(False)
        body(f,.204,'Each thin line joins fits to identical pixel counts. The thick line connects the means across datasets. Pairing exposes differences between the field models even when shot noise and calibration uncertainty produce substantial variation across runs. Panels use individual vertical scales.')
        body(f,.106,'With cross-coma, every seed favored the independent model laterally. Axial errors varied strongly with calibration. The paired interval on page 6 spans zero for axial RMSE, so this experiment does not resolve an axial advantage in that condition.')
        pdf.savefig(f); plt.close(f)
        f=page('Why line search stopped','Numerical audit at two reproduced failure points',8)
        original_runs=read_results('original_runs.csv')
        audit=read_results('line_search_audit.csv')
        body(f,.829,'The original solver proposed steps in float64, but rounded parameters to float32 before rendering intensities. Accumulating the final likelihood in float64 preserved that rounding. To test the effect, we evaluated the centered difference of the objective along the normalized projected descent direction and compared it with the autograd derivative at identical coordinates. The two examples use their original calibration or frozen field.')
        entries=[]
        for r in audit:
            if float(r['step'])>.001:
                continue
            entries.append([r['stage'].capitalize(),r['precision'].split('.')[-1],r['step'],
                            f"{float(r['finite_difference']):+.5f}",f"{float(r['autograd']):+.5f}"])
        table(f,[.08,.486,.84,.19],['Failed stage','Precision','Step','Difference','Autograd'],entries,
              [.22,.19,.15,.23,.21])
        body(f,.45,'At small steps the single-precision objective can report a rise along a descent direction. Both examples recover agreement when the pupil propagation and image formation use complex128 and the fit parameters and pixel arithmetic use float64. This check explains a failure mechanism. It does not establish that every original stop followed the same search trajectory.')
        def peak(key):
            return max(float(r[key]) for r in runs)
        n_original=sum(int(r[k])==2 for r in original_runs for k in ('cal_status','test_status'))
        body(f,.322,f'Fix. Keep the original fast fit as the initializer. Re-evaluate its solution with the high-precision renderer, then refine with bounded L-BFGS. Keep the original scaled projected-gradient target (0.003) and disable early stopping on a tiny relative objective change. The largest final gradient across calibration and localization was {max(peak("cal_pg"),peak("test_pg")):.6f}. The original {n_original} line-search stops fell to {statuses.count(2)}.')
        body(f,.176,'The data generator, seeds, likelihood, initialization, parameter bounds and model families are unchanged. Each test localization uses the newly refined calibration. A small projected gradient establishes constrained first-order stationarity at the returned solution. It cannot certify a unique pupil or an absolute axial origin.')
        pdf.savefig(f); plt.close(f)

        f=page('Did the axial result survive?','Same photon realizations and seeds before and after refinement  /  distances in nm',9)
        before={(r['scenario'],r['model']):r for r in read_results('original_summary.csv')}
        entries=[]
        for row in summary:
            old=before[row['scenario'],row['model']]
            entries.append([row['scenario'].replace(' field',''),row['model']]+
                           [f"{float(r[k]):+.2f}" if k=='z_bias' else f"{float(r[k]):.2f}"
                            for k in ('z_bias','z_rmse') for r in (old,row)])
        table(f,[.06,.65,.88,.17],['Truth','Fit','Bias old','Bias new','RMSE old','RMSE new'],entries,
              [.17,.25,.14,.14,.15,.15])
        body(f,.617,'Bias is the signed mean over held-out spots and calibration seeds. Large offsets with opposite signs can yield modest pooled bias. The table below splits axial mean-square error exactly into the square of each seed’s mean error and the remaining within-seed mean-square error, then averages over seeds. Square roots are reported.')
        entries=[]
        for row in summary:
            subset=[r for r in runs if (r['scenario'],r['model'])==(row['scenario'],row['model'])]
            offset=np.mean([float(r['z_bias'])**2 for r in subset])
            total=np.mean([float(r['z_rmse'])**2 for r in subset])
            entries.append([row['scenario'].replace(' field',''),row['model'],f'{np.sqrt(offset):.1f}',
                            f'{np.sqrt(total-offset):.1f}',f'{100*offset/total:.0f}%'])
        table(f,[.07,.323,.86,.175],['Truth','Fit','Offset RMS','Centered RMS','Offset share¹'],
              entries,[.16,.24,.20,.22,.18])
        body(f,.292,'¹ Fraction of axial mean-square error attributable to a single constant displacement per calibration. Removing that displacement uses test truth. Centered RMS describes relative-height accuracy and is unavailable as a correction for an ordinary blinking movie.')
        delta=max(abs(float(r[k])-float(before[r['scenario'],r['model']][k])) for r in summary for k in ('z_bias','z_rmse'))
        cross_intervals=[]
        for model in ('NAT','Independent','Independent + ridge'):
            signed=[float(r['z_bias']) for r in runs if r['scenario']=='Cross-coma field' and r['model']==model]
            low,high=np.percentile(rng.choice(signed,(10000,len(signed)),replace=True).mean(1),[2.5,97.5])
            cross_intervals.append(f'{low:+.1f} to {high:+.1f}')
        body(f,.192,f'The cross-coma mean signed bias has seed-bootstrap 95% intervals (nm) of {cross_intervals[0]} for NAT, {cross_intervals[1]} for independent, and {cross_intervals[2]} with ridge. The independent models have substantial between-calibration variability.')
        body(f,.103,f'The largest change among the six pooled biases and six RMSE values was {delta:.2f} nm. The tens-of-nm residual errors survive refinement. Pages 10–11 examine independent axial references.')
        pdf.savefig(f); plt.close(f)

        f=page('What transfers from bead fitting?','Separating numerical convergence from optical identifiability',10)
        body(f,.828,'How much does one unknown emitter height obscure optical defocus? For the reference pupil on page 3, we profiled out lateral position, flux and background, then compared the Poisson information for Noll 4 with z fixed versus z free. The other pupil modes were fixed. The equivalent fitted height shift below is the linearized response when an extra 0.010 waves of optical defocus in the data is omitted from the fit.')
        entries=[]
        for depth,(I,R,V) in zip([50,150,250],pixel_information()):
            free=I[0,0]
            known=free+R[2,0]**2/V[2,2]
            entries.append([str(depth),f'{.01*R[2,0]:+.1f}',f'{100*free/known:.2f}%'])
        table(f,[.08,.604,.84,.10],['Emitter height (nm)','Equivalent z shift (nm)','Information left with z free'],
              entries,[.29,.35,.36])
        body(f,.573,'1. Anchor the weak axial modes. The bead gel scan records nominal stage motion and fits residual drift with a reference origin. For in-situ data, carry over uncertainty on defocus from a nearby bead scan, or use repeated emitters at known focus steps with a common height per track. Compare absolute and relative z recovery on held-out reference emitters. A known step constrains relative motion. Absolute height still needs a reference.')
        body(f,.433,'2. Select field terms on unseen data. The planar bead estimator fits a global pupil first and accepts affine variation only if it predicts withheld exposures better. The in-situ pipeline already has checkerboard holdouts and multi-start z refits. Use those splits to compare NAT, NAT plus one cross-coma term, and the full field. Add depth-stratified residuals and a reference-height score alongside predictive likelihood. This nested model comparison remains to be tested.')
        body(f,.282,'3. Transfer optical constraints with their uncertainty. In the dense bead study, per-bead focus offsets beat a tilt plane on withheld clusters (deviance/pixel 1.393 vs 1.433). Keep per-emitter heights in situ and allow run-level focus variation if supported by independent reference frames. Profile per-frame brightness for blinking emitters. The stationary bead movie’s shared object assumption requires stable brightness.')
        body(f,.136,'Implementation sources: torch-psf 8b70122, study/bead_gel/README.md, docs/notes/dense_beads.md, estimation/movie_field.py and estimation/tracks.py. SIMCODE validation: simcode/pipeline/psf_fit.py. The in-situ track fitter uses Adam and already profiles brightness and background. This line-search repair applies to the comparison solver.')
        pdf.savefig(f); plt.close(f)
        f=page('What could axial references recover?','Random held-out reference splits of the simulated localizations',11)
        body(f,.832,'Take m of the 128 localized test spots as independent reference emitters with externally measured heights. Estimate one axial offset from their mean error, subtract it from the other 128 − m spots, and score those other spots only. The table averages squared error over every possible disjoint reference split and over all 12 calibration seeds. The field fit stays frozen.')
        entries=[]
        for row in summary:
            subset=[r for r in runs if (r['scenario'],r['model'])==(row['scenario'],row['model'])]
            centered=np.mean([float(r['z_rmse'])**2-float(r['z_bias'])**2 for r in subset])
            def anchored(m, reference_sd):
                return np.sqrt((128/127)*(1+1/m)*centered + reference_sd**2/m)
            entries.append([row['scenario'].replace(' field',''),row['model'],f'{float(row["z_rmse"]):.1f}',
                            f'{anchored(4,0):.1f}',f'{anchored(4,10):.1f}',f'{anchored(16,10):.1f}'])
        table(f,[.06,.531,.88,.171],['Truth','Fit','Original','4 exact','4 ±10 nm¹','16 ±10 nm¹'],
              entries,[.16,.26,.13,.13,.15,.17])
        body(f,.501,'Entries are axial RMSE in nm. ¹ The external height of each reference has independent, zero-mean error with SD 10 nm. In every column the localized reference positions themselves still contain the actual simulated single-molecule localization noise.')
        body(f,.382,'For N held-out sites, write s² for their within-dataset mean squared deviation about their mean error (denominator N). A random scored site and a random reference subset are disjoint. Their error difference has mean square [N/(N−1)] (1+1/m) s². Independent uncertainty in the known reference heights adds its variance divided by m. This identity uses the observed per-seed errors and introduces no fitting parameters.')
        body(f,.225,'This is a transfer experiment on simulated reference emitters, conditional on the same noise, optics and field coverage as the sample. A common systematic error in the reference height transfers directly to the corrected positions. Finite bead size, sample depth and registration must be accounted for when real beads supply the reference. A pure offset correction cannot remove spatially varying bias.')
        pdf.savefig(f); plt.close(f)
        f=page('How close is the fit to optimal?','Local Poisson precision on the same 96-spot calibration designs as the simulations',12)
        precision=read_results('matched_precision.csv')
        entries=[]
        for r in precision:
            obs=next((v for v in summary if v['scenario']==r['truth'] and v['model']==r['model']),None)
            entries.append([r['truth'].replace(' field',''),r['model'],f"{float(r['xy_local_sd_nm']):.2f}",
                            f"{float(obs['xy_sd']):.2f}" if obs else '—',
                            f"{float(r['z_local_sd_nm']):.2f}",f"{float(obs['z_sd']):.2f}" if obs else '—'])
        table(f,[.06,.653,.88,.154],['Truth','Fit','xy local','xy observed','z local','z observed'],entries,
              [.18,.21,.13,.17,.14,.17])
        body(f,.618,'Units are nm. The local calculation profiles out position, flux and background for each calibration spot at the true parameters. It propagates the shared pupil covariance to independent test spots and adds their photon noise. Variances are averaged across sites and seeds before taking square roots. Known-pupil rows remove calibration uncertainty. Earlier Fisher examples on pages 3–4 use a different design.')
        body(f,.476,'The constrained fit wins in z when the truth satisfies NAT. The local calculation predicts that advantage. Twelve repetitions, finite-sample bias and parameter bounds prevent an efficiency claim from the near agreement. The unbiased, correctly specified calculation does not apply to NAT under cross-coma or directly to a penalized fit.')
        body(f,.337,'Why did mean bias rise? On NAT-compatible data, seeds 1102, 1103 and 1107 shifted their mean z errors by +14.3, +19.6 and +13.4 nm after refinement. Their calibration objectives improved by only 0.048, 0.081 and 0.054 respectively. The other nine changed by less than 0.4 nm each. The optimizer moved along a weakly constrained axial direction. Its objective contains pixel counts, with no access to true emitter heights.')
        body(f,.192,'Precision here is the repeatability of emitter coordinates under repeated calibration and imaging. The comparison does not establish efficiency of all 13 global aberration coefficients: that would require the per-seed coefficient estimates and a full coefficient covariance comparison.')
        pdf.savefig(f); plt.close(f)

        f=page('How faithful is the NAT port?','Native Vectorfit field-function check  /  Python release 0.3.4',13)
        data=read_results('vectorfit_field_reference.csv')
        factors=np.array([PUPIL_NORM[j] for j in MODES])
        maxerr=0.
        for r in data:
            k=int(r['gamma'])-1
            ours=field_design(float(r['u']),float(r['v']))@nat_matrix()[:,k]*factors
            if k==8:
                ours/=np.sqrt(15/7)
            ref=np.array([float(r['Z'+str(j)]) for j in MODES])
            maxerr=max(maxerr,np.max(np.abs(ours-ref)))
        assert maxerr<1e-12
        body(f,.827,'We called the authors’ compiled compute_xy_gamma_legendre function using all 13 unit parameter vectors on a 5 × 5 normalized field grid and read all six mode coefficients. This independent check covers order, field axes, signs, zeros and shared terms. The conversion below compares the raw field polynomial outputs, before interpreting pupil normalization or optical path units.')
        body(f,.665,f'After reversing our per-mode pupil normalization, twelve columns agree directly. Column nine in Eq. S58 carries an extra √(15/7) relative to the binary. At (u,v) = (1,0), a unit g9 gives +1.5000 for raw Z6 in Vectorfit and +2.1958 in the supplement-based expansion. Dividing that column by √(15/7) gives a maximum absolute difference of {maxerr:.1e} across all 1,950 outputs.')
        body(f,.498,'Both descriptions span the same physical field shapes. A freely fitted g9 absorbs this scaling; a supplied coefficient or a prior expressed in those coordinates needs an explicit conversion. The simulation uses the paper convention consistently for generating and fitting. This source-level discrepancy therefore does not explain the change between its two optimizer runs.')
        body(f,.348,'The comparison uses our renderer and joint optimizer. The reference implementation alternates emitter and optical updates. Pixel predictions, physical units, the z origin/sign, derivatives and recovered estimates still require cross-renderer parity tests. A self-generated torch_psf simulation can test constraints internally; experimental validity requires independent data.')
        body(f,.192,'For a reference comparison, use one physical pupil and optical configuration, compare normalized pixels and derivatives through focus, then run both fitters on identical saved ROIs. The native Python package exposes the fitter, predicted pixels, derivatives and Fisher information. The published MATLAB demo remains a reference for the acquisition-to-fit workflow.')
        f.text(.08,.083,'Vectorfit Python release and authors’ demo instructions',color=teal,size=10,url='https://pypi.org/project/vectorfit-py/0.3.4/')
        pdf.savefig(f); plt.close(f)
        import json
        measured=json.loads(Path(__file__).with_name('dna_paint_fit.json').read_text())
        f=page('How close is NAT to the DNA-PAINT v4 PSF?', 'Experimental DNA-PAINT  /  two illumination axes',14)
        body(f,.826,'We sampled 2,670 isolated 15 × 15 spots across six acquisition windows and fitted each illumination axis separately. Alternating fits used camera-corrected intensities and per-pixel read variance. The NAT pupil started at zero aberration with the v4 optical geometry, focus and axis-specific dipole model. A 16-pixel checkerboard separated training and scoring locations.')
        entries=[[r['angle'],r['train'],r['held'],f"{r['mean_delta_nll']:+.2f}",f"{r['fraction_nat_better']:.1%}",f"{r['mean_z_shift_nm']:+.1f}"] for r in measured]
        table(f,[.075,.598,.85,.092],['Axis','Train','Held out','ΔNLL/ROI','NAT wins','Δz (nm)'],entries,[.1,.12,.15,.23,.2,.2])
        body(f,.571,'Δ is continuous NAT minus v4 after separate emitter refits. Lower NLL indicates better pixel prediction. The per-ROI standard errors of the paired differences are '+ ' and '.join(f"{r['se_delta_nll']:.2f}" for r in measured)+'. Spatial or temporal correlations can increase uncertainty. Δz measures a relative calibration shift. The acquisition has no reference heights. The v4 calibration used this movie.')
        table(f,[.075,.376,.85,.094],['Axis','0 nm','100 nm','200 nm','300 nm','400 nm'],
              [[r['angle']]+[f'{v:.1%}' for v in r['normalized_shape_relative_l2']] for r in measured],[.12,.16,.18,.18,.18,.18])
        body(f,.349,'Relative image L2 difference at equal xyz, normalized to unit ROI flux and averaged over 16 field positions. Training shared an integrated cubic spline per 60-pixel tile. On held-out ROIs, 30-pixel spline tiles gave RMS z differences of '+ ' and '.join(f"{r['spline_direct_z_rms_nm']:.1f} nm" for r in measured)+' versus the continuous NAT renderer for axes 0 and 1.')
        body(f,.192,'For in-situ fitting, use the spline for repeated emitter updates and rescore the chosen solution with the continuous pupil. Compare multiple initial pupils and inspect through-focus residuals before retraining. Independent reference heights or a registered bead stack would distinguish axial bias from a change of coordinate origin.')
        pdf.savefig(f); plt.close(f)
        f=page('Where does the spline lose accuracy?', 'DNA-PAINT  /  forward pixels at fixed position and fitted emitter heights',15)
        diagnostic=json.loads(Path(__file__).with_name('field_tile_errors.json').read_text())
        body(f,.827,'The spline cache shares one vectorial pupil across each 30-pixel field tile. We evaluated the resulting spline at each of the 16 tile centres, then 8 pixels away in both field coordinates in all four diagonal directions. Each comparison used the same subpixel position, five heights from 0 to 400 nm, and images normalized to unit flux.')
        table(f,[.075,.60,.85,.095],['Axis','At centre','Off centre','Off-centre max','Δz RMS'],
              [[r['angle'],f"{s['center']['relative_l2_mean']:.4%}",f"{s['offcenter']['relative_l2_mean']:.2%}",f"{s['offcenter']['relative_l2_max']:.1%}",f"{r['spline_direct_z_rms_nm']:.1f} nm"] for r,s in zip(measured,diagnostic)],
              [.10,.21,.23,.26,.2])
        body(f,.575,'The first three metrics are relative L2 pixel errors against the continuous pupil. The last column uses independently refitted experimental spots. The field approximation dominates fixed-position image error on this grid. Weak axial information, alternative localization minima and different lateral fit limits (3 versus 6.5 pixels) can also contribute to the height differences.')
        table(f,[.075,.389,.85,.092],['Axis','Direct time','Spline time','Speed ratio','Spline ΔNLL/ROI'],
              [[r['angle'],f"{r['seconds']['nat_direct']:.1f} s",f"{r['seconds']['nat_tiled_cold']:.1f} s",f"{r['seconds']['nat_direct']/r['seconds']['nat_tiled_cold']:.1f}×",f"{r['spline_minus_direct_nll']:+.2f}"] for r in measured],[.10,.2,.2,.21,.29])
        body(f,.362,'Timing includes lazy spline construction on a local RTX 4060 Laptop GPU. Positive ΔNLL means the tiled model fit the data worse. These measurements use three axial starts and 25 localization iterations. The 60-pixel training grid used ten alternating updates, each limited to three global steps. Every inner solve reached that budget.')
        polished=json.loads(Path(__file__).with_name('initialization_comparison.json').read_text())
        body(f,.204,'Matched-bound check. We sampled 48 held-out spots per axis and constrained both paths to ±3 pixels laterally. Direct refinement from the spline solution versus three direct starts gave RMS height differences of '+ ' and '.join(f"{r['polished_z_rms_nm']:.0f} nm" for r in polished)+'. Their mean direct NLL differences were '+ ' and '.join(f"{r['mean_polished_minus_cold_nll']:+.3f}" for r in polished)+' per spot. Both initializations won individual spots. Keep both starts when likelihood matters, and use independent axial references when judging height accuracy.')
        pdf.savefig(f); plt.close(f)
        f=page('Does the spline preserve the gradient?', 'Fixed emitter coordinates  /  all 13 NAT derivatives',16)
        gradients=json.loads(Path(__file__).with_name('gradient_check.json').read_text())
        body(f,.826,'Write K(g) for the prefiltered spline coefficients built from the optical model at NAT parameters g, and B for evaluation at fixed emitter coordinates. The predicted image is B K(g). The gradient of its likelihood is the transpose of the knot Jacobian applied to Bᵀ times the pixel likelihood gradient. Multiple emitters may accumulate into one knot adjoint before the optical backward pass.')
        table(f,[.075,.594,.85,.101],['Field point','Cached vs spline AD','Cached vs direct','Direction cosine¹'],
              [[('Tile centre' if r['offset_px_per_axis']==0 else '8 px on both axes'),f"{r['cached_vs_spline_autograd']:.2e}",f"{r['cached_vs_direct']:.2e}",f"{np.dot(r['direct'],r['cached'])/(np.linalg.norm(r['direct'])*np.linalg.norm(r['cached'])):.3f}"] for r in gradients],[.26,.25,.25,.24])
        body(f,.56,'Errors are Euclidean gradient differences divided by the reference gradient norm. ¹ Cosine between cached and direct 13-vectors. We used axis 0 of the fitted DNA-PAINT field, five fixed emitter heights, 2,000 photons, background 8, and a seeded Poisson draw at each of two positions. These are two numerical checks on synthetic counts, not uncertainty estimates for the DNA-PAINT experiment.')
        body(f,.395,'The cached backward pass agrees with automatic differentiation through the entire spline build. Agreement with the direct gradient requires agreement of the image and its parameter Jacobian. Even when pixel interpolation is accurate, rounding a field position to its tile centre changes both the pupil and the derivative of the pupil with respect to each NAT parameter.')
        body(f,.224,'To optimize the exact physical objective, use the spline for proposals or as a preconditioner and evaluate direct loss and gradient for the final refinement and stopping check. Identical stationary conditions still allow distinct local minima. Exact sharing is possible for identical evaluation geometry, or with a representation that exactly spans the direct forward model and its parameter derivatives.')
        pdf.savefig(f); plt.close(f)
        f=page('What does the speed ratio measure?', 'TorchPSF  /  integrated cubic B-splines, DNA-PAINT axis 0',17)
        b=json.loads(Path(__file__).with_name('spline_timing.json').read_text())
        body(f,.823,'The original 5.1× ratio on axis 0 compares whole localization jobs. It includes table construction for the spline, three axial starts, 25 optimizer iterations per start, camera variance and small field-tile batches. The vectorial path also runs on the GPU.')
        entries=[]
        for label,key in [('Render pixels','forward'),('Pixels and x, y, z derivatives','jacobian')]:
            direct, spl=b[f'vectorial_{key}_ms'],b[f'spline_{key}_ms']
            entries.append([label,f'{direct:.1f} ms',f'{spl:.1f} ms',f'{direct/spl:.1f}×'])
        table(f,[.075,.60,.85,.105],['Batch of 48 emitters','Vectorial','Cached spline','Ratio'],entries,[.43,.20,.23,.14])
        body(f,.568,'These warm timings use identical subpixel positions and a single physical pupil at the centre of one tile. Each entry is the median of five synchronized runs of ten calls on an RTX 4060 Laptop GPU. The Torch implementation integrates the spline over each pixel with separable tensor contractions. A fused evaluator or larger batch would require a separate benchmark.')
        body(f,.407,f"In a separate axis-0 rerun, building {b['occupied_tiles']} occupied tables took {b['build_s']:.1f} s. Localizing all {b['held']} spots with those tables already cached took {b['warm_fit_s']:.1f} s. The earlier cold timing was 36.9 s. Run-to-run variation precludes subtracting the timings to infer construction cost.")
        body(f,.236,'Retraining decision. The v4 trained estimator already uses an integrated cubic spline of the axis-specific calibration. Retain those weights while validating the NAT pupil. A paired frozen-weight test should use identical emitter coordinates, photon levels, backgrounds and camera settings under both pupils. Measure detection recall, axial bias and RMSE through depth and across field positions. Independent known-height data can establish which axial calibration transfers to the experiment.')
        pdf.savefig(f); plt.close(f)
if __name__ == '__main__':
    values = compare()
    for row in values:
        print(*row)
    report(values)
