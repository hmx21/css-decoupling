"""Native execution and rejection of removed CAS selection contracts."""
import builtins
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import css_decoupling
from css_decoupling import Ring, Matrix, decouple_general as decouple, decouple_complex
from css_decoupling.algebra.laurent import eye, zeros
from css_decoupling.algebra.modules import kernel, solve, using_backend
from css_decoupling.algebra.quillen_suslin import complete_polynomial_row
from css_decoupling.algebra.resolutions import image_resolution
from css_decoupling.certificates.construction import verify_construction
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.general_cli import main
from css_decoupling.models.examples import toric_css, shear_css
from css_decoupling.topology.complexes import direct_sum_complexes, disk


@contextmanager
def native_runtime_guard(monkeypatch):
    attempts = []
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    blocked = ('sage', 'sageall', 'sagemath', 'julia', 'juliacall', 'oscar',
               'singular', 'pysingular', 'cypari2',
               'css_decoupling.algebra.backends.singular',
               'css_decoupling.algebra.backends.process')

    def reject_module(name):
        name = name.lower()
        if any(name == prefix or name.startswith(prefix+'.') for prefix in blocked):
            attempts.append(('import', name))
            raise AssertionError('Native algorithms attempted an external CAS import: '+name)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        qualified = name
        if level and globals and globals.get('__package__'):
            qualified = importlib.util.resolve_name('.'*level+name, globals['__package__'])
        reject_module(qualified)
        for member in fromlist or ():
            if member != '*':
                reject_module(qualified+'.'+member)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        reject_module(importlib.util.resolve_name(name, package) if name.startswith('.') else name)
        return original_import_module(name, package)

    def external(*args, **kwargs):
        attempts.append(('process-or-discovery', args[0] if args else None))
        raise AssertionError('Native algorithms attempted an external program')

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', guarded_import)
        patch.setattr(importlib, 'import_module', guarded_import_module)
        patch.setattr(subprocess, 'Popen', external)
        patch.setattr(os, 'system', external)
        patch.setattr(os, 'popen', external)
        patch.setattr(shutil, 'which', external)
        yield
    assert not attempts, 'External attempts were suppressed by the called algorithm'


def test_qs_resolution_and_raw_pipeline_run_under_native_runtime_guard(monkeypatch):
    with native_runtime_guard(monkeypatch):
        R = Ring(5, 2)
        x, y = R.variables
        row = Matrix.from_rows(R, [[y*y, 1+x*y, x*x-x]])
        trace = []
        U, V = complete_polynomial_row(row, timeout=20, max_terms=100000, trace=trace)
        assert row*U == Matrix.from_rows(R, [[1, 0, 0]])
        assert U*V == V*U == eye(R, 3)
        assert trace and len(trace[0]['patch_denominators']) == 2

        A = Matrix.from_rows(R, [[x**-2*(x-1), y-1, x+y-2, 0]])
        maps = image_resolution(A, method='native', timeout=20, max_terms=100000)
        assert maps[0]*solve(maps[0], A) == A
        assert A*solve(A, maps[0]) == maps[0]
        for previous, following in zip(maps, maps[1:]):
            assert not previous*following
            K = kernel(previous)
            assert following*solve(following, K) == K
        assert not kernel(maps[-1])

        R3 = Ring(3, 3)
        HX, HZ = toric_css(R3, {1: 1})
        HX, HZ = shear_css(HX, HZ, 0, 2, 2*R3.variable(0)**-1+R3.variable(2))
        result = decouple(HX, HZ, backend='sympy', resolution_method='native',
                          resolution_timeout=20, max_cell_index=8)
        assert result.multiplicities == {1: 1}
        assert verify_decoupling(result.certificate()).valid


def test_removed_cas_modules_are_not_distributed_or_importable():
    package = Path(css_decoupling.__file__).resolve().parent
    for leaf in ('singular', 'process'):
        assert not (package/'algebra'/'backends'/(leaf+'.py')).exists()
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('css_decoupling.algebra.backends.'+leaf)


@pytest.mark.parametrize('zero', [False, True])
def test_removed_resolution_options_are_rejected_even_for_empty_work(zero):
    R = Ring(3, 2)
    A = zeros(R, 2, 3) if zero else eye(R, 2)
    with pytest.raises(ValueError, match='native'):
        image_resolution(A, method='singular')
    with pytest.raises(TypeError, match='executable'):
        image_resolution(A, executable='Singular')


def test_removed_backend_api_selections_are_rejected():
    with pytest.raises(ValueError, match='backend'):
        with using_backend('singular'):
            pytest.fail('Removed backend was selected')
    HX, HZ = toric_css(Ring(3, 2), {1: 1})
    for option in ({'backend': 'singular'}, {'resolution_method': 'singular'},
                   {'linear_backend': 'singular'}):
        with pytest.raises(ValueError):
            decouple(HX, HZ, **option)


def test_css_completion_rejects_removed_resolution_before_any_kernel(monkeypatch):
    from css_decoupling.topology import resolutions

    def forbidden(*args, **kwargs):
        pytest.fail('Removed resolution selection reached native kernel computation')

    monkeypatch.setattr(resolutions, 'kernel', forbidden)
    HX, HZ = toric_css(Ring(3, 2), {1: 1})
    with pytest.raises(ValueError, match='native'):
        resolutions.css_complex(HX, HZ, resolution_method='singular')


@pytest.mark.parametrize('command,option', [
    ('run', '--backend'), ('general', '--backend'), ('analyze', '--backend'),
    ('coarse', '--backend'), ('verify', '--backend'), ('torus', '--backend'),
    ('lattice-check', '--backend'), ('run', '--resolution-method'),
    ('analyze', '--resolution-method'), ('coarse', '--resolution-method'),
    ('lattice-check', '--resolution-method'), ('run', '--linear-backend'),
    ('construct', '--linear-backend'),
])
def test_cli_rejects_removed_cas_choice_before_reading_input(command, option, tmp_path, capsys):
    args = [command, str(tmp_path/'missing-input.json')]
    if command != 'verify':
        args += ['--output', str(tmp_path/'must-not-exist.json')]
    if command in ('torus', 'lattice-check'):
        args += ['--basis', '[[1,0],[0,1]]']
    with pytest.raises(SystemExit) as error:
        main(args+[option, 'singular'])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    assert not (tmp_path/'must-not-exist.json').exists()


def test_legacy_backend_label_is_provenance_and_never_selects_a_cas(monkeypatch):
    with native_runtime_guard(monkeypatch):
        R = Ring(3, 2)
        C = direct_sum_complexes(disk(R, -1), disk(R, 2))
        result = decouple_complex(C, linear_backend='sympy', contraction_strategy='sections')
        old_label = deepcopy(result.certificate())
        old_label['configuration']['linear_backend'] = 'singular'
        report = verify_construction(old_label)
        assert report.valid, report.errors
        with pytest.raises(ValueError):
            verify_construction(old_label, linear_backend='singular')
