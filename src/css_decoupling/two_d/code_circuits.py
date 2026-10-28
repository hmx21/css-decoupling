"""Decoupling circuits on the original register, with target-code freedom.

The supplied fixed-rank factorization is U=S*F. Target correction G makes
A=G*F a circuit to the same standard code, without adding physical sites.
K=G*S^-1 is certified as a standard-code automorphism, so A=K*U need not
equal the initial QCA U. This is not the stable Higman circuit contract.
"""
from __future__ import annotations

from .algebra import Matrix, Ring, eye, solve, NoLaurentSolution
from .circuits import CircuitResourceLimit, _apply_gate_rows, _validate_gate
from .direct_circuits import (
    compile_direct_circuit, verify_direct_circuit, _equal_metadata,
)
from .models import standard_code
from .verify import verify, verify_generation


SCHEMA = "css-decoupling-2d/no-input-ancilla/v1"
_KEYS = {"schema", "field_characteristic", "dimension", "qca_result",
         "initial_factorization", "correction", "actual_circuit",
         "target_automorphism", "code_binding", "equivalence", "resources",
         "construction"}
_ACTION_KEYS = {"z_action", "z_inverse", "x_action", "x_inverse", "layers"}
_CONVENTIONS = {
    "matrix_action": "columns; chronological gates multiply on the left",
    "physical_Z": "z_action", "physical_X": "z_inverse dagger",
    "pure_X_Z_phase": 0,
    "MULTIPLY": "|a> -> |multiplier*a>; Z exponent /multiplier; X exponent *multiplier",
}
_IDENTITY_METHOD = "low-rank-identity-after-bidirectional-stabilizer-check"
_IDENTITY_REASON = "q<3 and original coarse checks generate exactly the standard target stabilizer modules"
_SCOPE = "same-register code decoupling; no added input ancillas"


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _terms(A):
    return sum(len(f.terms) for f in A.entries.values())


def _bounded(A, cap):
    if _terms(A) > cap:
        raise CircuitResourceLimit("No-input-ancilla matrix monomial budget exceeded")
    return A


def _matrix(A, ring, q, cap):
    if not isinstance(A, Matrix) or A.ring != ring or A.shape != (q,q):
        raise ValueError("Physical action has the wrong ring or original register size")
    return _bounded(A,cap)


def _inverse_pair(A, B, ring, q, cap):
    _matrix(A,ring,q,cap);_matrix(B,ring,q,cap)
    if q > cap:
        raise CircuitResourceLimit("Original register exceeds identity monomial budget")
    I=eye(ring,q)
    if _bounded(A*B,cap) != I or _bounded(B*A,cap) != I:
        raise ValueError("Physical actions fail their two-sided inverse identities")


def _gate_species(ring, q, gate):
    if isinstance(gate,dict) and gate.get("kind") == "MULTIPLY":
        if set(gate) != {"kind","qudit","multiplier"}:
            raise ValueError("Malformed on-site multiplication gate")
        site=_integer(gate["qudit"],"qudit")
        value=gate["multiplier"]
        if site >= q or type(value) is not int or not 0 < value < ring.p:
            raise ValueError("Invalid multiplication site or canonical field multiplier")
        return {site}
    _validate_gate(ring,q,gate)
    return {gate["control"],gate["target"]}


def code_gate_product(ring, physical_qudits, layers, *, pauli="Z", reverse=False,
                      max_families=200000, max_matrix_terms=1000000):
    """Replay actual SUM and on-site multiplication gates; no factors consulted."""
    if not isinstance(ring,Ring) or ring.D != 2:
        raise ValueError("A two-variable prime Laurent ring is required")
    q=_integer(physical_qudits,"physical_qudits")
    family_cap=_integer(max_families,"max_families")
    cap=_integer(max_matrix_terms,"max_matrix_terms")
    if pauli not in ("X","Z") or type(reverse) is not bool:
        raise ValueError("Invalid physical gate replay mode")
    if not isinstance(layers,list):
        raise ValueError("Gate layers must be a list")
    families=0
    for layer in layers:
        if not isinstance(layer,list) or not layer:
            raise ValueError("Gate layers must be nonempty lists")
        occupied=set()
        for gate in layer:
            species=_gate_species(ring,q,gate)
            if occupied & species:
                raise ValueError("Overlapping physical gate families in one layer")
            occupied.update(species);families+=1
            if families > family_cap:
                raise CircuitResourceLimit("No-input-ancilla gate-family budget exceeded")
    if q > cap:
        raise CircuitResourceLimit("Original register exceeds identity monomial budget")
    rows=[{i:ring.one()} for i in range(q)]
    retained=q
    for layer in reversed(layers) if reverse else layers:
        for gate in reversed(layer) if reverse else layer:
            if gate["kind"] == "MULTIPLY":
                a=gate["multiplier"]
                exponent=(-1 if pauli == "Z" else 1)*(-1 if reverse else 1)
                scalar=pow(a,exponent,ring.p)
                site=gate["qudit"]
                rows[site]={j:scalar*f for j,f in rows[site].items()}
            else:
                retained+=_apply_gate_rows(ring,rows,gate,pauli,reverse=reverse)
            if retained > cap:
                raise CircuitResourceLimit("No-input-ancilla gate replay monomial budget exceeded")
    return Matrix(ring,q,q,{(i,j):f for i,row in enumerate(rows) for j,f in row.items()})


def _geometric(ring, axis, n, cap):
    """The exact Laurent polynomial (x_axis**n-1)/(x_axis-1)."""
    if abs(n) > cap:
        raise CircuitResourceLimit("Toric shift correction exceeds its monomial budget")
    powers=range(n) if n >= 0 else range(n,0)
    coefficient=1 if n >= 0 else -1
    return ring({tuple(k if j==axis else 0 for j in range(2)):coefficient for k in powers})


def _actions(Z, Zinv, layers):
    return {"z_action":Z,"z_inverse":Zinv,"x_action":Zinv.dagger(),
            "x_inverse":Z.dagger(),"layers":layers}


def target_transport_correction(standard_HX, standard_HZ, toric_copies,
                                product_x, product_z, transport, *,
                                max_families=200000,max_matrix_terms=1000000):
    """Remove diagonal transport using existing target species only.

    An existing product species absorbs all toric units using elementary
    diag(u,u^-1) pairs. Without products, each toric pair uses paper VI.3;
    scalar units are implemented by explicit on-site multiplication gates.
    """
    t=_integer(toric_copies,"toric_copies")
    px=_integer(product_x,"product_x");pz=_integer(product_z,"product_z")
    cap=_integer(max_matrix_terms,"max_matrix_terms")
    family_cap=_integer(max_families,"max_families")
    ring=standard_HX.ring;q=px+pz+2*t
    TX,TZ=standard_code(t,product_x=px,product_z=pz,ring=ring)
    if standard_HX != TX or standard_HZ != TZ:
        raise ValueError("Transport correction requires the canonical toric/product target")
    S=_matrix(transport,ring,q,cap)
    if any(i!=j for i,j in S.entries) or any(not S[i,i].is_unit for i in range(q)):
        raise ValueError("Target transport must be diagonal Laurent units")
    Sinv=Matrix(ring,q,q,{(i,i):S[i,i]**-1 for i in range(q)})
    layers=[]

    def emit(gate):
        if len(layers) >= family_cap:
            raise CircuitResourceLimit("Target correction gate-family budget exceeded")
        layers.append([gate])

    def elementary(i,j,f):
        if len(f.terms) > cap:
            raise CircuitResourceLimit("Target correction coefficient budget exceeded")
        for e,c in sorted(f.terms.items()):
            emit({"kind":"CNOT" if ring.p==2 else "SUM","control":i,"target":j,
                  "offset":list(e),"coefficient":-c % ring.p})

    if px+pz:
        # Anchor 0 is already a target product species; no site is introduced.
        for i in range(px+pz,q):
            u=S[i,i]
            if u==1:
                continue
            for first,second,f in ((i,0,-ring.one()),(0,i,ring.one()),(i,0,-ring.one()),
                                   (i,0,u),(0,i,-u**-1),(i,0,u)):
                elementary(first,second,f)
        strategy="existing-target-product"
    else:
        x,y=ring.variables
        for k in range(t):
            i,j=2*k,2*k+1
            ((a,b),c1),=S[i,i].terms.items()
            ((c,d),c2),=S[j,j].terms.items()
            # G=R*C: physical scalar gates occur before the toric shears.
            for site,coefficient in ((i,c1),(j,c2)):
                if coefficient != 1:
                    emit({"kind":"MULTIPLY","qudit":site,
                          "multiplier":pow(coefficient,-1,ring.p)})
            f=_bounded_poly((y-1)*_geometric(ring,0,c-a,cap),cap)
            g=_bounded_poly(x**(a-c)*(x-1)*_geometric(ring,1,b-d,cap),cap)
            # Written R=E_ij(f) E_ji(g), hence this chronological order.
            elementary(j,i,g);elementary(i,j,f)
        strategy="toric-pairs"
    G=code_gate_product(ring,q,layers,max_families=family_cap,max_matrix_terms=cap)
    Ginv=code_gate_product(ring,q,layers,reverse=True,max_families=family_cap,max_matrix_terms=cap)
    K,Kinv=_bounded(G*Sinv,cap),_bounded(S*Ginv,cap)
    DX,DZ=eye(ring,px+t),eye(ring,pz+t)
    for i in range(px):
        DX[i,i]=(K[i,i]**-1).involution()
    for i in range(pz):
        DZ[i,i]=K[px+i,px+i]
    if not px+pz:
        for k in range(t):
            ((a,b),_),=S[2*k,2*k].terms.items()
            ((c,d),_),=S[2*k+1,2*k+1].terms.items()
            DX[k,k]=ring.monomial((-a,-d))
            DZ[k,k]=ring.monomial((-c,-b))
    DXinv=Matrix(ring,DX.nrows,DX.ncols,{(i,i):DX[i,i]**-1 for i in range(DX.nrows)})
    DZinv=Matrix(ring,DZ.nrows,DZ.ncols,{(i,i):DZ[i,i]**-1 for i in range(DZ.nrows)})
    automorphism={"z_action":K,"z_inverse":Kinv,"x_action":Kinv.dagger(),
                  "x_inverse":K.dagger(),"stabilizers":{
                      "X":{"source_in_target":DX,"target_in_source":DXinv},
                      "Z":{"source_in_target":DZ,"target_in_source":DZinv}}}
    _verify_target_automorphism(automorphism,TX,TZ,cap)
    return {"strategy":strategy,**_actions(G,Ginv,layers)},automorphism


def _bounded_poly(f, cap):
    if len(f.terms)>cap:
        raise CircuitResourceLimit("Toric correction polynomial budget exceeded")
    return f


def _verify_target_automorphism(data, TX, TZ, cap):
    if not isinstance(data,dict) or set(data)!={"z_action","z_inverse","x_action","x_inverse","stabilizers"}:
        raise ValueError("Malformed target-code automorphism")
    ring,q=TX.ring,TX.ncols
    K,Kinv=data["z_action"],data["z_inverse"]
    _inverse_pair(K,Kinv,ring,q,cap)
    if data["x_action"]!=Kinv.dagger() or data["x_inverse"]!=K.dagger():
        raise ValueError("Target automorphism has an invalid physical X action")
    if set(data["stabilizers"])!={"X","Z"}:
        raise ValueError("Both target stabilizer sectors require witnesses")
    for sector,A,B,H in (("Z",K,Kinv,TZ),("X",Kinv.dagger(),K.dagger(),TX)):
        witness=data["stabilizers"][sector]
        if set(witness)!={"source_in_target","target_in_source"}:
            raise ValueError("Malformed target automorphism generation witnesses")
        forward,backward=witness["source_in_target"],witness["target_in_source"]
        _inverse_pair(forward,backward,ring,H.nrows,cap)
        if (_bounded(A*H.dagger(),cap)!=H.dagger()*forward or
                _bounded(B*H.dagger(),cap)!=H.dagger()*backward):
            raise ValueError("The proposed correction does not preserve the standard target code")


def _resources(q, px, pz, layers):
    total=sum(len(layer) for layer in layers)
    multipliers=sum(g["kind"]=="MULTIPLY" for layer in layers for g in layer)
    offsets=[g["offset"] for layer in layers for g in layer if g["kind"]!="MULTIPLY"]
    return {"input_ancillas":0,"physical_qudits":q,"output_physical_qudits":q,
            "target_product_X":px,"target_product_Z":pz,"depth":len(layers),
            "gate_family_count":total,"SUM_family_count":total-multipliers,
            "MULTIPLY_family_count":multipliers,
            "range_l1_in_supercell_coordinates":max((sum(map(abs,e)) for e in offsets),default=0),
            "range_linf_in_supercell_coordinates":max((max(map(abs,e)) for e in offsets),default=0)}


def compile_code_circuit(result, factor_data, *, verify_source=True,
                         max_factors=200000,max_families=200000,
                         max_matrix_terms=1000000,progress=None):
    """Compile supplied original-rank factors; no factorizer is called here."""
    if type(verify_source) is not bool:
        raise ValueError("verify_source must be boolean")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    cap=_integer(max_matrix_terms,"max_matrix_terms")
    family_cap=_integer(max_families,"max_families")
    if verify_source:
        verify(result,max_families=family_cap,max_matrix_terms=cap)
    if not isinstance(factor_data,dict) or type(factor_data.get("added_ancillas")) is not int or factor_data["added_ancillas"]!=0:
        raise ValueError("A no-input-ancilla circuit requires an original-rank factorization")
    Q=result["qca"];U,V=Q["psi"][1],Q["psi_inverse"][1]
    ring,q=U.ring,U.nrows
    initial=compile_direct_circuit(U,V,factor_data["factors"],
        shift_exponents=[list(e) for e in factor_data["shift_exponents"]],
        transport_coefficients=factor_data.get("transport_coefficients"),
        max_factors=max_factors,max_families=family_cap,max_matrix_terms=cap)
    TX,TZ=Q["standard_HX"],Q["standard_HZ"]
    px,pz,t=Q["product_x"],Q["product_z"],Q["toric_copies"]
    if progress:progress("target_transport_correction",{"physical_qudits":q,"input_ancillas":0})
    used=initial["resources"]["gate_family_count"]
    correction,automorphism=target_transport_correction(TX,TZ,t,px,pz,initial["transport"]["z_action"],
        max_families=family_cap-used,max_matrix_terms=cap)
    G,Ginv=correction["z_action"],correction["z_inverse"]
    A,Ainv=_bounded(G*initial["z_action"],cap),_bounded(initial["z_inverse"]*Ginv,cap)
    layers=initial["layers"]+correction["layers"]
    old=result["qca_stabilizers"];changes=automorphism["stabilizers"]
    stabilizers={sector:{
        "source_in_target":changes[sector]["source_in_target"]*old[sector]["source_in_target"],
        "target_in_source":old[sector]["target_in_source"]*changes[sector]["target_in_source"]}
        for sector in ("X","Z")}
    artifact={"schema":SCHEMA,"field_characteristic":ring.p,"dimension":2,
        "qca_result":result,"initial_factorization":initial,"correction":correction,
        "actual_circuit":{**_actions(A,Ainv,layers),"conventions":dict(_CONVENTIONS)},
        "target_automorphism":automorphism,
        "code_binding":{"source_HX":result["coarse_original_HX"],"source_HZ":result["coarse_original_HZ"],
                        "target_HX":TX,"target_HZ":TZ,"stabilizers":stabilizers,
                        "retained_lattice":result["lattice"]["basis_columns"]},
        "equivalence":{"input_ancillas":0,"physical_qudits":q,"stabilizer_equivalent":True,
                       "original_qca_implemented_by_circuit":A==U,
                       "equation":"actual_z = target_automorphism_z * initial_qca"},
        "resources":_resources(q,px,pz,layers),
        "construction":{"method":factor_data.get("algorithm","supplied-original-rank-factors"),
                        "scope":_SCOPE}}
    # The source and initial core have just been verified; validate the new
    # correction, actual gates and original-code binding without redoing them.
    _verify_code_result(artifact,max_families=family_cap,max_matrix_terms=cap)
    return artifact


def compile_qca_no_ancilla(result, *, factorization="auto", factor_data=None,
                           timeout=120,max_steps=1000000,max_matrix_terms=1000000,
                           max_factors=200000,max_families=200000,max_products=30000000,
                           max_patches=128,max_field_trials=128,seed=0,
                           verify_source=True,progress=None,rank_two_options=None):
    """Factor the initial q-by-q QCA and correct its target transport in place."""
    if factorization not in ("auto","suslin","fast"):
        raise ValueError("Unknown original-rank factorization strategy")
    if type(verify_source) is not bool:
        raise ValueError("verify_source must be boolean")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    for name,value in (("max_factors",max_factors),("max_families",max_families),
                       ("max_matrix_terms",max_matrix_terms)):
        _integer(value,name)
    for name,value in (("max_steps",max_steps),("max_products",max_products),
                       ("max_patches",max_patches),("max_field_trials",max_field_trials)):
        _integer(value,name,1)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(timeout) not in (int,float) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")
    if rank_two_options is not None:
        from .rank_two import parse_rank_two_options
        rank_two_options=parse_rank_two_options(rank_two_options)
    if verify_source:
        verify(result,max_families=max_families,max_matrix_terms=max_matrix_terms)
    # This is a code-specific equality check, not a rank-two elementary
    # factorization claim and not a fallback that swallows factorizer errors.
    if factor_data is None and result["qca"]["psi"][1].nrows < 3:
        identity_result=_low_rank_identity(result,max_matrix_terms=max_matrix_terms,
                                            max_families=max_families)
        if identity_result is not None:
            if progress:progress("low_rank_identical_code",{"input_ancillas":0,
                "physical_qudits":identity_result["resources"]["physical_qudits"]})
            return identity_result
    rank_two_factorization=False
    if factor_data is None:
        Q=result["qca"]
        if Q["psi"][1].nrows==2 and factorization=="auto":
            from .rank_two import factor_rank_two
            factor_data=factor_rank_two(result,timeout=timeout,max_steps=max_steps,
                max_terms=max_matrix_terms,max_factors=max_factors,max_products=max_products,
                rank_two_options=rank_two_options,progress=progress)
            rank_two_factorization=True
        else:
            from .unstable import factor_unstabilized
            factor_data=factor_unstabilized(Q["psi"][1],Q["psi_inverse"][1],method=factorization,
                timeout=timeout,max_steps=max_steps,max_terms=max_matrix_terms,
                max_factors=max_factors,max_products=max_products,max_patches=max_patches,
                max_field_trials=max_field_trials,seed=seed,progress=progress)
    try:
        return compile_code_circuit(result,factor_data,verify_source=False,max_factors=max_factors,
            max_families=max_families,max_matrix_terms=max_matrix_terms,progress=progress)
    except CircuitResourceLimit as exc:
        if not rank_two_factorization:
            raise
        from .rank_two_budget import RankTwoResourceLimit
        raise RankTwoResourceLimit('circuit_resource_limit',str(exc),
            phase='rank_two_gate_compilation',details={
                'general_suslin_rank_requirement':'q >= 3','non_elementarity_proved':False,
                'limits':{'max_families':max_families,'max_matrix_terms':max_matrix_terms,
                          'max_factors':max_factors},
                'factorization':factor_data.get('statistics',{})}) from exc


def _low_rank_identity(source, *, max_matrix_terms, max_families):
    cap=_integer(max_matrix_terms,"max_matrix_terms")
    Q=source["qca"];U,V=Q["psi"][1],Q["psi_inverse"][1]
    ring,q=U.ring,U.nrows
    if q >= 3:
        return None
    TX,TZ=Q["standard_HX"],Q["standard_HZ"]
    OX,OZ=source["coarse_original_HX"],source["coarse_original_HZ"]
    changes={}
    try:
        for sector,old,target in (("X",OX,TX),("Z",OZ,TZ)):
            _bounded(old,cap);_bounded(target,cap)
            changes[sector]={"source_in_target":solve(target.dagger(),old.dagger()),
                             "target_in_source":solve(old.dagger(),target.dagger())}
    except NoLaurentSolution:
        return None
    I=eye(ring,q)
    auto_changes={sector:{
        "source_in_target":changes[sector]["source_in_target"]*source["qca_stabilizers"][sector]["target_in_source"],
        "target_in_source":source["qca_stabilizers"][sector]["source_in_target"]*changes[sector]["target_in_source"]}
        for sector in ("X","Z")}
    artifact={"schema":SCHEMA,"field_characteristic":ring.p,"dimension":2,"qca_result":source,
        "initial_factorization":None,"correction":None,
        "actual_circuit":{**_actions(I,I,[]),"conventions":dict(_CONVENTIONS)},
        "target_automorphism":{"z_action":V.copy(),"z_inverse":U.copy(),
            "x_action":U.dagger(),"x_inverse":V.dagger(),"stabilizers":auto_changes},
        "code_binding":{"source_HX":OX,"source_HZ":OZ,"target_HX":TX,"target_HZ":TZ,
                        "stabilizers":changes,"retained_lattice":source["lattice"]["basis_columns"]},
        "equivalence":{"input_ancillas":0,"physical_qudits":q,"stabilizer_equivalent":True,
                       "original_qca_implemented_by_circuit":I==U,
                       "equation":"actual_z = target_automorphism_z * initial_qca"},
        "resources":_resources(q,Q["product_x"],Q["product_z"],[]),
        "construction":{"method":_IDENTITY_METHOD,"reason":_IDENTITY_REASON,"scope":_SCOPE}}
    _verify_code_result(artifact,max_families=max_families,max_matrix_terms=cap)
    return artifact


def _verify_low_rank_identity(artifact, cap):
    source=artifact["qca_result"];Q=source["qca"]
    U,V=Q["psi"][1],Q["psi_inverse"][1];ring,q=U.ring,U.nrows
    if q >= 3 or artifact["correction"] is not None:
        raise ValueError("The identity branch requires fewer than three original physical species")
    if not _equal_metadata(artifact["construction"],{
            "method":_IDENTITY_METHOD,"reason":_IDENTITY_REASON,"scope":_SCOPE}):
        raise ValueError("The identity branch lacks its exact code-specific justification")
    if q > cap:
        raise CircuitResourceLimit("Identity physical register exceeds the monomial budget")
    I=eye(ring,q)
    if not _equal_metadata(artifact["actual_circuit"],{**_actions(I,I,[]),"conventions":dict(_CONVENTIONS)}):
        raise ValueError("The identity branch must have actual identity action and no gates")
    TX,TZ=Q["standard_HX"],Q["standard_HZ"]
    auto=artifact["target_automorphism"]
    _verify_target_automorphism(auto,TX,TZ,cap)
    if auto["z_action"]!=V or auto["z_inverse"]!=U:
        raise ValueError("Identity code equivalence must retain the inverse of the initial QCA")
    binding=artifact["code_binding"]
    expected={"source_HX":source["coarse_original_HX"],"source_HZ":source["coarse_original_HZ"],
              "target_HX":TX,"target_HZ":TZ,"retained_lattice":source["lattice"]["basis_columns"]}
    if set(binding)!=set(expected)|{"stabilizers"} or not _equal_metadata({k:binding[k] for k in expected},expected):
        raise ValueError("Identity code source, standard target or lattice mismatch")
    if set(binding["stabilizers"])!={"X","Z"}:
        raise ValueError("Both identity-code stabilizer sectors require witnesses")
    # These are both inclusions for the actual input and target codes, not
    # tests of a finite torus or dimensions/ranks alone.
    verify_generation(binding["source_HX"].dagger(),TX.dagger(),binding["stabilizers"]["X"])
    verify_generation(binding["source_HZ"].dagger(),TZ.dagger(),binding["stabilizers"]["Z"])
    expected_equivalence={"input_ancillas":0,"physical_qudits":q,"stabilizer_equivalent":True,
        "original_qca_implemented_by_circuit":I==U,
        "equation":"actual_z = target_automorphism_z * initial_qca"}
    if not _equal_metadata(artifact["equivalence"],expected_equivalence):
        raise ValueError("Identity code equivalence declaration differs")
    resources=_resources(q,Q["product_x"],Q["product_z"],[])
    if not _equal_metadata(artifact["resources"],resources):
        raise ValueError("Identity code resource declaration differs")
    return {"verified":True,"qca":True,"circuit":True,"input_ancillas":0,
            "physical_qudits":q,"stabilizer_equivalent":True,
            "original_qca_implemented_by_circuit":I==U,"resources":resources,
            "target_product_X":Q["product_x"],"target_product_Z":Q["product_z"]}


def _verify_code_result(artifact, *, max_families=200000,max_matrix_terms=1000000):
    cap=_integer(max_matrix_terms,"max_matrix_terms")
    family_cap=_integer(max_families,"max_families")
    if not isinstance(artifact,dict) or set(artifact)!=_KEYS or artifact["schema"]!=SCHEMA:
        raise ValueError("Malformed no-input-ancilla certificate")
    source=artifact["qca_result"];Q=source["qca"]
    U,V=Q["psi"][1],Q["psi_inverse"][1];ring,q=U.ring,U.nrows
    if not _equal_metadata([artifact["field_characteristic"],artifact["dimension"]],[ring.p,2]):
        raise ValueError("No-input-ancilla field metadata differs")
    initial=artifact["initial_factorization"]
    if initial is None:
        return _verify_low_rank_identity(artifact,cap)
    if (initial["original_qca"]!=U or initial["original_inverse"]!=V or
            initial["ancillas"]["count"]!=0 or initial["z_action"].shape!=(q,q)):
        raise ValueError("Initial factorization changed the physical register or QCA")
    TX,TZ=Q["standard_HX"],Q["standard_HZ"]
    correction=artifact["correction"]
    if set(correction)!=_ACTION_KEYS|{"strategy"} or correction["strategy"] not in ("existing-target-product","toric-pairs"):
        raise ValueError("Malformed target transport correction")
    G,Ginv=correction["z_action"],correction["z_inverse"]
    _inverse_pair(G,Ginv,ring,q,cap)
    if correction["x_action"]!=Ginv.dagger() or correction["x_inverse"]!=G.dagger():
        raise ValueError("Correction Pauli action differs")
    # Independently replay all four physical correction streams. The initial
    # core reader checks the initial stream; exact concatenation below proves
    # the total stream without rereading its potentially large initial prefix.
    for pauli,reverse,key in (("Z",False,"z_action"),("Z",True,"z_inverse"),
                              ("X",False,"x_action"),("X",True,"x_inverse")):
        if code_gate_product(ring,q,correction["layers"],pauli=pauli,reverse=reverse,
                max_families=family_cap,max_matrix_terms=cap)!=correction[key]:
            raise ValueError("Actual correction gates differ from their physical Pauli action")
    auto=artifact["target_automorphism"]
    _verify_target_automorphism(auto,TX,TZ,cap)
    S,Sinv=initial["transport"]["z_action"],initial["transport"]["z_inverse"]
    if auto["z_action"]!=_bounded(G*Sinv,cap) or auto["z_inverse"]!=_bounded(S*Ginv,cap):
        raise ValueError("Target automorphism does not relate the correction to initial transport")
    actual=artifact["actual_circuit"]
    if set(actual)!=_ACTION_KEYS|{"conventions"} or not _equal_metadata(actual["conventions"],_CONVENTIONS):
        raise ValueError("Malformed actual circuit or phase convention")
    A,Ainv=actual["z_action"],actual["z_inverse"]
    _inverse_pair(A,Ainv,ring,q,cap)
    if (A!=_bounded(G*initial["z_action"],cap) or Ainv!=_bounded(initial["z_inverse"]*Ginv,cap) or
            A!=_bounded(auto["z_action"]*U,cap) or Ainv!=_bounded(V*auto["z_inverse"],cap)):
        raise ValueError("Actual circuit is not the claimed same-code replacement of the initial QCA")
    if actual["x_action"]!=Ainv.dagger() or actual["x_inverse"]!=A.dagger():
        raise ValueError("Actual circuit has an invalid physical X action")
    if not _equal_metadata(actual["layers"],initial["layers"]+correction["layers"]):
        raise ValueError("Actual gate stream is not the chronological initial stream and correction")
    if sum(len(layer) for layer in actual["layers"])>family_cap:
        raise CircuitResourceLimit("No-input-ancilla total gate-family budget exceeded")
    binding=artifact["code_binding"]
    expected={"source_HX":source["coarse_original_HX"],"source_HZ":source["coarse_original_HZ"],
              "target_HX":TX,"target_HZ":TZ,"retained_lattice":source["lattice"]["basis_columns"]}
    if set(binding)!=set(expected)|{"stabilizers"} or not _equal_metadata({k:binding[k] for k in expected},expected):
        raise ValueError("Original checks, canonical target or retained lattice mismatch")
    if set(binding["stabilizers"])!={"X","Z"}:
        raise ValueError("Both code stabilizer sectors require witnesses")
    verify_generation(A*binding["source_HZ"].dagger(),TZ.dagger(),binding["stabilizers"]["Z"])
    verify_generation(Ainv.dagger()*binding["source_HX"].dagger(),TX.dagger(),binding["stabilizers"]["X"])
    expected_equivalence={"input_ancillas":0,"physical_qudits":q,"stabilizer_equivalent":True,
        "original_qca_implemented_by_circuit":A==U,
        "equation":"actual_z = target_automorphism_z * initial_qca"}
    if not _equal_metadata(artifact["equivalence"],expected_equivalence):
        raise ValueError("Incorrect input-ancilla, register or QCA-equivalence declaration")
    expected_resources=_resources(q,Q["product_x"],Q["product_z"],actual["layers"])
    if not _equal_metadata(artifact["resources"],expected_resources):
        raise ValueError("No-input-ancilla gate resources mismatch")
    construction=artifact["construction"]
    if (not isinstance(construction,dict) or set(construction)!={"method","scope"} or
            type(construction["method"]) is not str or not construction["method"] or
            construction["scope"]!=_SCOPE):
        raise ValueError("Malformed construction scope")
    return {"verified":True,"qca":True,"circuit":True,"input_ancillas":0,
            "physical_qudits":q,"stabilizer_equivalent":True,
            "original_qca_implemented_by_circuit":A==U,
            "target_product_X":Q["product_x"],"target_product_Z":Q["product_z"],
            "resources":expected_resources}


def verify_no_ancilla_result(artifact, *, expected_HX=None,expected_HZ=None,
                            max_factors=200000,max_families=200000,max_matrix_terms=1000000):
    """Independent full reader: no factorizer or correction constructor is called."""
    if not isinstance(artifact,dict) or artifact.get("schema")!=SCHEMA:
        raise ValueError("Unsupported no-input-ancilla certificate")
    _integer(max_factors,"max_factors")
    report=verify(artifact["qca_result"],expected_HX=expected_HX,expected_HZ=expected_HZ,
                  max_families=max_families,max_matrix_terms=max_matrix_terms)
    if artifact["initial_factorization"] is not None:
        verify_direct_circuit(artifact["initial_factorization"],max_factors=max_factors,
                              max_families=max_families,max_matrix_terms=max_matrix_terms)
    return {**report,**_verify_code_result(artifact,max_families=max_families,max_matrix_terms=max_matrix_terms)}
