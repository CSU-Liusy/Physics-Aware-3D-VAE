"""Run-owned legacy paths; input directories are always read-only."""
from pathlib import Path
import hashlib
import json

REPOSITORY_ROOT=Path(__file__).resolve().parents[1]


def external_input(path):
    if path is None: raise ValueError('Supply an explicit external legacy input path')
    result=Path(path).expanduser().resolve()
    if result==REPOSITORY_ROOT or result.is_relative_to(REPOSITORY_ROOT):
        raise ValueError('Legacy mesh inputs and checkpoints must be outside the public repository')
    return result


def select_mesh_files(ply_dir, file_list=None):
    """Select exactly the explicit file(s), or the PLY files in one directory."""
    if isinstance(file_list,(list,tuple)):
        candidates=[external_input(p) for p in file_list]
    elif file_list is not None:
        selected=external_input(file_list)
        candidates=[p for p in selected.iterdir() if p.suffix.lower()=='.ply'] if selected.is_dir() else [selected]
    else:
        directory=external_input(ply_dir)
        if not directory.is_dir(): raise FileNotFoundError(f'PLY input directory does not exist: {directory}')
        candidates=[p for p in directory.iterdir() if p.suffix.lower()=='.ply']
    if any(not p.is_file() or p.suffix.lower()!='.ply' for p in candidates):
        raise ValueError('Every explicitly selected input must be an existing PLY file')
    files=sorted({str(p.resolve()) for p in candidates})
    if not files: raise ValueError('No PLY input files selected')
    return files


def resolve_output_directory(output, input_directories):
    if output is None: raise ValueError('Supply an explicit external output directory for legacy work')
    target=Path(output).expanduser().resolve()
    if target==REPOSITORY_ROOT or target.is_relative_to(REPOSITORY_ROOT):
        raise ValueError('Legacy outputs and caches must be outside the public repository')
    for directory in input_directories:
        source=external_input(directory)
        if source.is_file(): source=source.parent
        if target==source or target.is_relative_to(source):
            raise ValueError('Legacy outputs/cache must be outside every input directory')
    return target


def prepare_mesh_cache(cache_root, source_files, grid_size, reset=False):
    sources=[external_input(p) for p in source_files]
    if not sources: raise ValueError('No source mesh supplied for cache identity')
    if len({p.name for p in sources})!=len(sources):
        raise ValueError('Legacy mesh cache requires distinct source basenames')
    root=resolve_output_directory(cache_root,[p.parent for p in sources])
    identity={'format':'legacy-mesh-cache-v1','grid_size':list(grid_size),'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':')).encode()
    namespace=root/hashlib.sha256(encoded).hexdigest()
    if namespace.is_symlink(): raise ValueError('Cache namespace must not be a symlink')
    marker=namespace/'cache_identity.json'
    if namespace.exists():
        if not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text(encoding='utf-8'))!=identity:
            raise ValueError('Existing cache has no matching legacy identity; refusing to modify it')
    else:
        namespace.mkdir(parents=True)
        marker.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    if reset:
        d,h,w=grid_size
        for source in sources:
            cached=namespace/f'{source.name}.vox{d}x{h}x{w}.npy'
            if cached.is_symlink(): raise ValueError('Refusing a symlink in the run-owned mesh cache')
            if cached.resolve().parent!=namespace.resolve(): raise ValueError('Cache target escaped its identity namespace')
            if cached.is_file(): cached.unlink()
    return str(namespace)
