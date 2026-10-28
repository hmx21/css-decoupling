"""Strict, lossless JSON for finite Laurent data. No pickle, eval or CAS parser."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

from .algebra import Matrix, Ring, Poly


def integer(value, name="integer", minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}")
    return value


def encode(value):
    if isinstance(value, Matrix):
        return {"__matrix__": 1, "p": value.ring.p, "D": value.ring.D,
                "shape": list(value.shape), "entries": [
                    [i, j, [[*e, c] for e, c in sorted(f.terms.items())]]
                    for (i, j), f in sorted(value.entries.items())]}
    if isinstance(value, dict):
        # Integer keys occur only in degree-indexed maps; tag them explicitly.
        if any(type(k) is int for k in value):
            if not all(type(k) is int for k in value):
                raise TypeError("Mixed mapping key types")
            return {"__integer_mapping__": [[k, encode(v)] for k, v in sorted(value.items())]}
        if not all(type(k) is str for k in value):
            raise TypeError("Invalid mapping key")
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__css_matrix_table__" in value:
        return _decode_table(value)
    if "__matrix_ref__" in value:
        raise ValueError("Matrix reference outside a matrix table")
    if "__matrix__" in value:
        if set(value) != {"__matrix__", "p", "D", "shape", "entries"} or type(value["__matrix__"]) is not int or value["__matrix__"] != 1:
            raise ValueError("Malformed matrix record")
        R = Ring(integer(value["p"], "prime", 2), integer(value["D"], "dimension", 1))
        if not isinstance(value["shape"], list) or len(value["shape"]) != 2:
            raise ValueError("Invalid matrix shape")
        nr, nc = [integer(v, "matrix dimension", 0) for v in value["shape"]]
        entries = {}
        if not isinstance(value["entries"], list):
            raise ValueError("Matrix entries must be a list")
        for item in value["entries"]:
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError("Malformed matrix entry")
            i, j, terms = item
            integer(i, "row", 0); integer(j, "column", 0)
            if i >= nr or j >= nc or (i, j) in entries or not isinstance(terms, list) or not terms:
                raise ValueError("Noncanonical matrix position")
            coefficients = {}
            for term in terms:
                if not isinstance(term, list) or len(term) != R.D + 1:
                    raise ValueError("Malformed Laurent term")
                exponent = tuple(integer(e, "exponent") for e in term[:-1])
                c = integer(term[-1], "coefficient", 1)
                if c >= R.p or exponent in coefficients:
                    raise ValueError("Noncanonical Laurent coefficient")
                coefficients[exponent] = c
            entries[i, j] = Poly(R, coefficients)
        return Matrix(R, nr, nc, entries)
    if "__integer_mapping__" in value:
        if set(value) != {"__integer_mapping__"}:
            raise ValueError("Malformed degree mapping")
        out = {}
        for key, item in value["__integer_mapping__"]:
            integer(key, "degree")
            if key in out:
                raise ValueError("Duplicate degree")
            out[key] = decode(item)
        return out
    return {k: decode(v) for k, v in value.items()}


def canonical_bytes(value):
    return (json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _unique_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("Duplicate JSON key")
        out[k] = v
    return out


def _table_payload(value):
    """Deduplicate by exact matrix equality, never by a hash alone.

    Keep existing non-matrix lists/dicts when possible, so a large gate stream
    is not copied just to replace a few matrix values by table references.
    """
    matrices=[];buckets={};identities={}
    def visit(item):
        if isinstance(item,Matrix):
            index=identities.get(id(item))
            if index is None:
                fingerprint=(item.ring,item.shape,sum(hash((i,j,f)) for (i,j),f in item.entries.items()))
                candidates=buckets.setdefault(fingerprint,[])
                index=next((k for k in candidates if matrices[k] == item),None)
                if index is None:
                    index=len(matrices);matrices.append(item);candidates.append(index)
                identities[id(item)]=index
            return {"__matrix_ref__":index}
        if isinstance(item,dict):
            if any(type(k) is int for k in item):
                if not all(type(k) is int for k in item):raise TypeError("Mixed mapping key types")
                return {"__integer_mapping__":[[k,visit(v)] for k,v in sorted(item.items())]}
            if not all(type(k) is str for k in item):raise TypeError("Invalid mapping key")
            if any(k in item for k in ("__css_matrix_table__","__matrix_ref__")):
                raise ValueError("Reserved matrix transport key")
            out={k:visit(v) for k,v in sorted(item.items())}
            return item if all(out[k] is v for k,v in item.items()) else out
        if isinstance(item,(tuple,list)):
            out=[visit(v) for v in item]
            return item if isinstance(item,list) and all(a is b for a,b in zip(out,item)) else out
        if item is None or type(item) in (str,int,bool,float):return item
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    return visit(value),matrices


def _decode_table(record, *, consume=False):
    if (set(record)!={"__css_matrix_table__","matrices","payload"} or
        type(record["__css_matrix_table__"]) is not int or record["__css_matrix_table__"] != 1 or
        not isinstance(record["matrices"],list)):
        raise ValueError("Malformed matrix-table transport")
    matrices=[]
    for position,entry in enumerate(record["matrices"]):
        if not isinstance(entry,dict) or "__matrix__" not in entry:
            raise ValueError("Matrix table contains a non-matrix")
        matrices.append(decode(entry))
        if consume:
            record["matrices"][position]=None
    entry=None
    used=set()
    def visit(item):
        if isinstance(item,list):
            if consume:
                for i in range(len(item)):item[i]=visit(item[i])
                return item
            return [visit(v) for v in item]
        if not isinstance(item,dict):return item
        if "__matrix_ref__" in item:
            if set(item)!={"__matrix_ref__"}:raise ValueError("Malformed matrix reference")
            index=integer(item["__matrix_ref__"],"matrix reference",0)
            if index>=len(matrices):raise ValueError("Matrix reference out of range")
            used.add(index);return matrices[index]
        if any(k in item for k in ("__matrix__","__css_matrix_table__")):
            raise ValueError("Nested matrix transport or inline matrix in table payload")
        if "__integer_mapping__" in item:
            if set(item)!={"__integer_mapping__"}:raise ValueError("Malformed degree mapping")
            out={}
            for key,value in item["__integer_mapping__"]:
                integer(key,"degree")
                if key in out:raise ValueError("Duplicate degree")
                out[key]=visit(value)
            return out
        if consume:
            for k in item:item[k]=visit(item[k])
            return item
        return {k:visit(v) for k,v in item.items()}
    result=visit(record["payload"])
    if used != set(range(len(matrices))):raise ValueError("Unreferenced matrix-table entry")
    return result


def save(value, path, *, compact=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted serialization leaves an explicit .partial, never an empty
    # apparent certificate. Hard-link publication is atomic and refuses an
    # existing destination, including a race with another writer.
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+".partial")
    payload,matrices=_table_payload(value) if compact else (encode(value),[])
    encoder=json.JSONEncoder(sort_keys=True,separators=(",",":"),allow_nan=False)
    with temporary.open("x",encoding="utf8",newline="\n") as stream:
        def write(item):
            # Use CPython's C encoder for one record at a time. The outer
            # matrix-table stream still avoids constructing a JSON tree or
            # string for all matrices together; Python token-by-token JSON
            # iteration was a measurable I/O bottleneck on large certificates.
            stream.write(encoder.encode(item))
        if matrices:
            stream.write('{"__css_matrix_table__":1,"matrices":[')
            for i,matrix in enumerate(matrices):
                if i:stream.write(',')
                write(encode(matrix))
            stream.write('],"payload":');write(payload);stream.write('}\n')
        else:
            write(payload);stream.write('\n')
        stream.flush()
    os.link(temporary,path)
    temporary.unlink()


def load(path):
    def finite_float(text):
        value=float(text)
        if not math.isfinite(value):raise ValueError("Nonfinite JSON number")
        return value
    raw=json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=_unique_pairs,
                   parse_float=finite_float,
                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    # The parser's private tree has no external aliases. Consume table records
    # as they become exact native matrices, rather than retaining two complete
    # representations of a large certificate throughout conversion.
    if isinstance(raw,dict) and "__css_matrix_table__" in raw:
        return _decode_table(raw,consume=True)
    return decode(raw)
