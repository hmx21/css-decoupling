"""Publish separate QCA/circuit JSON with a shared complete source certificate.

Files are exclusively created, never overwritten. Caught publication failures
remove only files created by this call. Arbitrary output directories do not
provide a cross-file crash transaction: readers must verify all source bindings.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import stat


def output_paths(certificate_path, *, qca_path=None, circuit_path=None):
    """Default sidecars are based on the complete certificate's filename stem."""
    path = Path(certificate_path)
    return {
        "certificate": path,
        "qca": Path(qca_path) if qca_path is not None else path.with_name(path.stem + ".qca.json"),
        "circuit": Path(circuit_path) if circuit_path is not None else path.with_name(path.stem + ".circuit.json"),
    }


def check_new_output_paths(paths, *, protected=()):
    """Resolve aliases and reject input/output collisions before algebra or writes."""
    expanded = {name: Path(path).expanduser() for name, path in paths.items()}
    for path in expanded.values():
        # Resolve the parent, not the final entry: resolving a dangling final
        # symlink would hide the fact that an output entry already exists.
        final_entry = path.parent.resolve() / path.name
        if path.is_symlink() or final_entry.is_symlink():
            raise FileExistsError("Refusing an output symlink: " + str(path))
    resolved = {name: path.resolve() for name, path in expanded.items()}
    keys = [os.path.normcase(str(path)) for path in resolved.values()]
    if len(set(keys)) != len(keys):
        raise ValueError("Output files must have distinct paths")
    protected_keys = {os.path.normcase(str(Path(path).expanduser().resolve())) for path in protected}
    if protected_keys.intersection(keys):
        raise ValueError("Output path overlaps an input or source certificate")
    for path in resolved.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError("Refusing to overwrite existing output: " + str(path))
    return resolved


def write_new_json_files(payloads, paths, *, protected=()):
    """Prepare every JSON first, then publish in mapping order with rollback."""
    if set(payloads) != set(paths):
        raise ValueError("Output payloads and paths must have identical keys")
    resolved = check_new_output_paths(paths, protected=protected)
    encoded = {name: json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
               for name, value in payloads.items()}
    created = []
    try:
        for name, path in resolved.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                identity = os.fstat(stream.fileno())
                created.append((path, identity.st_dev, identity.st_ino))
                stream.write(encoded[name])
    except BaseException as original:
        cleanup_failures = []
        for path, device, inode in reversed(created):
            try:
                actual = path.lstat()
                # A different process may have replaced the path. Never remove
                # a replacement which this call did not create.
                if not stat.S_ISLNK(actual.st_mode) and (actual.st_dev, actual.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_failures.append(str(path))
        if cleanup_failures:
            raise OSError("Output publication failed; could not remove newly created files: "
                          + ", ".join(cleanup_failures)) from original
        raise
    return resolved


def _source(value):
    return deepcopy(value.certificate() if hasattr(value, "certificate") else value)


def save_decoupling_outputs(result, certificate_path, *, qca_path=None, circuit_path=None,
                            linear_backend=None, bounded_options=None, max_total_rank=100000):
    """Verify a complete result and save its certificate and two bound sidecars.

    Old v1 certificate fields are unchanged. The complete certificate is
    published last; a crash-created orphan sidecar is not a completed bundle.
    """
    from .outputs import export_outputs
    planned = output_paths(certificate_path, qca_path=qca_path, circuit_path=circuit_path)
    check_new_output_paths(planned)
    source = _source(result)
    exported = export_outputs(source, linear_backend=linear_backend,
                              bounded_options=bounded_options, max_total_rank=max_total_rank)
    paths = {name: planned[name] for name in ("qca", "circuit", "certificate")}
    return write_new_json_files({**exported, "certificate": source}, paths)


def save_exported_outputs(source_certificate, *, qca_path, circuit_path,
                          linear_backend=None, bounded_options=None, max_total_rank=100000,
                          source_path=None):
    """Export an existing complete certificate without rewriting that source."""
    from .outputs import export_outputs
    paths = {"qca": qca_path, "circuit": circuit_path}
    protected = () if source_path is None else (source_path,)
    check_new_output_paths(paths, protected=protected)
    source = _source(source_certificate)
    exported = export_outputs(source, linear_backend=linear_backend,
                              bounded_options=bounded_options, max_total_rank=max_total_rank)
    return write_new_json_files(exported, paths, protected=protected)
