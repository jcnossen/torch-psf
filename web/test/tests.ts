import { runAll } from './harness';
import { getDevice, describeGpu } from '../src/gpu/device';
import './test_forward';
import './test_nat';
import './test_field_spline';
import './test_dense_length';
import './test_cloud';
import './test_tiff';
import './test_fit';
import './test_export';
import './test_psfzip';
import './test_pdf';
import './test_bench';
import './test_smlm';
import './test_insitu_vjp';

getDevice().then(({ info }) => runAll(describeGpu(info)), (e) => runAll('none: ' + e.message));
