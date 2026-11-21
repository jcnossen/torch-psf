"""Model extensions: psf.zip / psf.json entries whose code lives outside torch-psf (format 0.7, docs/PSF_FORMAT.md).

A model block (the dict a model's _file_dict writes and its _from_file reads; for 4Pi also the FourPiPSF state) may
carry
    "extensions": [{"name": "mylab.custom_detection", "version": 1, "required": true, "state": {...}}]
The file only names the extension and stores its parameters. The code comes from a package installed in the reading
environment, never from the file: register_extension(name, load, version) when that package is imported, or a
`torch_psf.extensions` entry point named after the extension (pyproject.toml:
[project.entry-points."torch_psf.extensions"] "mylab.custom_detection" = "mylab.psf_detection:load").

Reading a model applies its entries in order: load(model, state, version) -> the extended model; a model that is a
torch.nn.Module is then moved to the device it was read to (a loader need not place new modules itself; for other
models the loader must create its tensors on model.dev). An extension that is
not installed (or whose installed loader is older than the entry's version) is an error when the entry is required
(the default) and a warning when it is optional (the base model is returned without it). Names are namespaced by the
providing package ("<package>.<feature>"); names starting with "torch_psf." are reserved.

Writing: a model reports its entries with file_extensions() (PSF default: none). The writers put them into the model
block (unless the block already holds them, as the FourPiPSF state does) and raise the file version to 0.7.
"""
import warnings

import torch

EXTENSIONS_VERSION = (0, 7)
ENTRY_POINT_GROUP = 'torch_psf.extensions'
_registry = {}          # name -> (load, version)


class ExtensionError(ValueError):
    """A required model extension is not available in this environment."""


def register_extension(name, load, version=1):
    """Make `load(model, state, version) -> model` the reader of extension `name` for entry versions <= `version`."""
    if not isinstance(name, str) or not name or '.' not in name:
        raise ValueError("extension names are namespaced: '<package>.<feature>'")
    if name.startswith('torch_psf.'):
        raise ValueError("extension names starting with 'torch_psf.' are reserved")
    if not callable(load):
        raise TypeError('load must be callable: load(model, state, version) -> model')
    _registry[name] = (load, int(version))
    return load


def unregister_extension(name):
    _registry.pop(name, None)


def registered_extensions():
    """{name: version} of the extensions this environment can read (entry points not yet imported are not listed)."""
    return {k: v for k, (_, v) in _registry.items()}


def entry(name, state, version=1, required=True):
    """The file entry of an extension: what a model's file_extensions() returns, one per extension."""
    return dict(name=str(name), version=int(version), required=bool(required), state=state)


def _from_entry_points(name):
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:        # broken metadata of an unrelated package: no discovery, the registry still works
        return None
    for ep in eps:
        if ep.name == name:
            obj = ep.load()
            if name not in _registry and callable(obj):
                register_extension(name, obj, version=getattr(obj, 'extension_version', 1))
            return _registry.get(name)
    return None


def _lookup(name):
    return _registry.get(name) or _from_entry_points(name)


def names_of(model):
    fn = getattr(model, 'file_extensions', None)
    return {e['name'] for e in (fn() if fn is not None else [])}


def apply(model, entries, where='PSF model'):
    """Apply the extension entries of a model block to the model read from it; returns the (possibly new) model.
    Entries the model already carries (by name, e.g. applied while reading an embedded state) are skipped."""
    for e in entries or []:
        name, version = e.get('name'), int(e.get('version', 1))
        required = bool(e.get('required', True))
        if name in names_of(model):
            continue
        found = _lookup(name)
        problem = None
        if found is None:
            problem = f'needs the extension {name!r} (version {version}), which is not installed'
        elif found[1] < version:
            problem = f'needs the extension {name!r} version {version}; the installed one reads up to {found[1]}'
        if problem is not None:
            hint = ('; install / import the package that provides it (it registers itself with '
                    'torch_psf.extensions.register_extension or a "torch_psf.extensions" entry point)')
            if required:
                raise ExtensionError(f'{where} {problem}{hint}')
            warnings.warn(f'{where} {problem}: loaded without it (optional extension)')
            continue
        dev = getattr(model, 'dev', None)
        model = found[0](model, e.get('state'), version)
        if dev is not None and isinstance(model, torch.nn.Module):
            model = model.to(dev)   # modules the loader created (often on the CPU) to the device the model was read to
    return model


def used(block):
    """True if a (nested) psf.json block holds a non-empty 'extensions' list."""
    if isinstance(block, dict):
        return bool(block.get('extensions')) or any(used(v) for v in block.values())
    if isinstance(block, (list, tuple)):
        return any(used(v) for v in block)
    return False


def with_entries(model, d):
    """The model block d plus the model's extension entries (unchanged when it has none or d already holds them)."""
    fn = getattr(model, 'file_extensions', None)
    ex = list(fn()) if fn is not None else []
    if not ex or used(d):
        return d
    return dict(d, extensions=ex)
