import { runAll } from './harness';
import { getDevice, describeGpu } from '../src/gpu/device';
import './test_forward';
import './test_tiff';
import './test_fit';
import './test_export';
import './test_bench';

getDevice().then(({ info }) => runAll(describeGpu(info)), (e) => runAll('none: ' + e.message));
