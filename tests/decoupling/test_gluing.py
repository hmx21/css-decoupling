from copy import deepcopy
from itertools import combinations

import pytest

from css_decoupling import Ring, Matrix, zeros
from css_decoupling.algebra.modules import using_backend
from css_decoupling.errors import NoLaurentSolution
from css_decoupling.models.examples import glued_css, glued_toric_complex, glued_toric_chain
from css_decoupling.topology.complexes import direct_sum_complexes, koszul
from css_decoupling.topology.cohomology import profile
from css_decoupling.decoupling.gluing import gluing_classes, killing_axes
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.certificates.gluing import verify_gluing
from css_decoupling.certificates.analysis import verify_analysis


@pytest.mark.parametrize('D,i,j', [(3,1,2),(4,1,3),(5,2,3),(5,2,4)])
def test_signed_koszul_cocycle_in_known_layer_bases(D,i,j):
    ring=Ring(5,D)
    C=glued_toric_complex(ring,i,j,coefficient=2)
    lower=koszul(ring,i)
    low=zeros(ring,C.rank(i),1); low[0,0]=1
    high=zeros(ring,C.rank(j),1); high[lower.rank(j),0]=1
    # These are fixed geometric Koszul bases, independent of GB basis choices.
    step=gluing_classes(C,{i:low,j:high},j)
    order=j-i+1
    sign=-(-1)**((order+1)*i)
    assert step.classes[0].terms()==((tuple(range(order)),0,0,(2*sign)%5),)
    assert killing_axes(step.classes)==(0,)
    assert verify_gluing(step.certificate()).valid


def test_unsplit_lower_tower_and_omitted_layer_are_rejected():
    ring=Ring(2,4)
    C=glued_toric_chain(ring,[1,2,3])
    reps={j:l.representatives for j,l in profile(C).items()}
    with pytest.raises(NoLaurentSolution):
        gluing_classes(C,reps,3)
    C=direct_sum_complexes(glued_toric_complex(ring,1,2),koszul(ring,3))
    reps={j:l.representatives for j,l in profile(C).items()}
    with pytest.raises(ValueError,match='every nonzero layer'):
        gluing_classes(C,{1:reps[1],3:reps[3]},3)


def test_zero_class_and_multiple_cocycle_supports():
    ring=Ring(3,4)
    C=glued_toric_complex(ring,coefficient=0)
    step=gluing_classes(C,{j:l.representatives for j,l in profile(C).items()},2)
    assert step.is_zero and killing_axes(step.classes)==()
    C=direct_sum_complexes(glued_toric_complex(ring,axes=(0,1)),
                          glued_toric_complex(ring,axes=(2,3)))
    step=gluing_classes(C,{j:l.representatives for j,l in profile(C).items()},2)
    assert killing_axes(step.classes)==(0,2)
    assert verify_gluing(step.certificate()).valid


@pytest.mark.parametrize('prime',[2,3])
def test_cocycle_guided_blocking_and_bound_analysis_certificate(prime):
    ring=Ring(prime,3)
    result=analyze_profile(*glued_css(ring))
    assert result.translation_kernel.index==1
    assert result.lattice.index==prime  # uniform old fallback was prime**3
    assert [r['step'].is_zero for r in result.gluing]==[False,True]
    data=result.certificate()
    assert verify_analysis(data).valid
    for mutate in (
        lambda d:d['attachments'][0].update(kill_axes=[]),
        lambda d:d['attachments'][0]['basis'][0].__setitem__(0,True),
        lambda d:d['final_basis'][0].__setitem__(0,True),
        lambda d:d['attachments'][0]['gluing']['classes'][0]['coefficients'].update(entries=[]),
        lambda d:d['input']['HX'].update(entries=[]),
    ):
        bad=deepcopy(data); mutate(bad)
        assert not verify_analysis(bad).valid


def test_exact_mixed_translation_relations_reduce_cell_before_gluing():
    ring=Ring(3,2)
    x,y=ring.variables
    HX=Matrix.from_rows(ring,[[x+1,y+1]])
    HZ=Matrix.from_rows(ring,[[-(y+1)],[x+1]]).dagger()
    result=analyze_profile(HX,HZ)
    assert result.translation_kernel.index==2  # old axiswise result was 4
    assert result.lattice.index==2 and result.gluing==[]
    assert verify_analysis(result.certificate()).valid


def test_supplied_lattice_failure_exposes_actual_gluing_witness():
    with pytest.raises(NoLaurentSolution) as failure:
        analyze_profile(*glued_css(Ring(3,3)),lattice_basis=[[1,0,0],[0,1,0],[0,0,1]])
    assert verify_gluing(failure.value.gluing_certificate).valid
