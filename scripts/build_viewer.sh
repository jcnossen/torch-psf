#!/bin/sh
# Bundle the PSF viewer front end into the Python package (run before building a wheel / sdist, or once in a source
# checkout): builds the viewer entry of the web/ project and copies it to torch_psf/viewer_static/ (git-ignored),
# which `python -m torch_psf view` serves at /.  Needs Node.js (npm).
set -e
cd "$(dirname "$0")/.."
npm --prefix web ci
npm --prefix web run build:viewer
rm -rf torch_psf/viewer_static
cp -r web/dist-viewer torch_psf/viewer_static
echo "viewer front end bundled in torch_psf/viewer_static"
