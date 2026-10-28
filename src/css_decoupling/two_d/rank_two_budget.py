"""Shared, cooperative budgets and structured rank-two incomplete outcomes."""
from contextlib import contextmanager
from time import perf_counter

from .algebra import Matrix, Poly
from ..errors import ResourceLimit
from .unstable import UnstableFactorizationIncomplete


class RankTwoIncomplete(UnstableFactorizationIncomplete):
    """A bounded rank-two route did not produce a certificate, not nonmembership."""
    def __init__(self, reason, message, *, phase, details):
        super().__init__(message)
        self.reason, self.phase, self.details = reason, phase, details


class RankTwoResourceLimit(ResourceLimit):
    def __init__(self, reason, message, *, phase, details):
        super().__init__(message)
        self.reason, self.phase, self.details = reason, phase, details


class ProbeStopped(Exception):
    """The optional fast probe used its local share; global budgets still apply."""


def natural(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


class RankTwoBudget:
    def __init__(self, *, timeout, max_steps, max_terms, max_products, max_factors):
        if type(timeout) not in (int, float) or not 0 < timeout < float('inf'):
            raise ValueError('timeout must be finite and positive')
        self.started = perf_counter()
        self.deadline = self.started+timeout
        self.max_steps = natural(max_steps, 'max_steps', minimum=1)
        self.max_terms = natural(max_terms, 'max_terms')
        self.max_products = natural(max_products, 'max_products', minimum=1)
        self.max_factors = natural(max_factors, 'max_factors')
        self.steps = self.products = self.nodes = 0
        self.phase = 'rank_two_validation'
        self._probe = None

    def snapshot(self):
        return {'operations': self.steps, 'convolution_pairs': self.products,
                'generated_nodes': self.nodes, 'elapsed_seconds': perf_counter()-self.started,
                'limits': {'max_steps': self.max_steps, 'max_products': self.max_products,
                           'max_terms': self.max_terms, 'max_factors': self.max_factors}}

    def stop(self, reason, message, **details):
        raise RankTwoResourceLimit(reason, message, phase=self.phase,
                                   details={**details, 'budget': self.snapshot(),
                                            'general_suslin_rank_requirement': 'q >= 3',
                                            'non_elementarity_proved': False})

    def incomplete(self, reason, message, **details):
        raise RankTwoIncomplete(reason, message, phase=self.phase,
                                details={**details, 'budget': self.snapshot(),
                                         'general_suslin_rank_requirement': 'q >= 3',
                                         'non_elementarity_proved': False})

    def check(self, *values, terms=0):
        self.steps += 1
        now = perf_counter()
        if now > self.deadline:
            self.stop('timeout', 'Rank-two shared cooperative time budget exhausted')
        if self.steps > self.max_steps:
            self.stop('step_limit', 'Rank-two shared operation budget exhausted')
        if self.products > self.max_products:
            self.stop('product_limit', 'Rank-two shared convolution-product budget exhausted')
        for value in values:
            if isinstance(value, Matrix):
                terms += sum(len(f.terms) for f in value.entries.values())
            elif isinstance(value, Poly):
                terms += len(value.terms)
        if terms > self.max_terms:
            self.stop('term_limit', 'Rank-two retained support budget exhausted', retained_terms=terms)
        if self._probe is not None:
            steps, products, deadline = self._probe
            if self.steps > steps or self.products > products or now > deadline:
                raise ProbeStopped()

    def charge(self, pairs):
        self.products += pairs
        self.check()

    def multiply(self, a, b):
        self.check(a, b)
        self.charge(len(a.terms)*len(b.terms))
        value = a*b
        self.check(value)
        return value

    def mm(self, A, B):
        self.check(A, B)
        rows = {}
        for (i, j), f in B.entries.items():
            rows.setdefault(i, []).append(f)
        self.charge(sum(len(f.terms)*len(g.terms)
                        for (_, k), f in A.entries.items() for g in rows.get(k, ())))
        result = A*B
        self.check(result)
        return result

    def scale(self, A, coefficient):
        result = Matrix(A.ring, A.nrows, A.ncols,
                        {at: self.multiply(f, coefficient) for at, f in A.entries.items()})
        self.check(result)
        return result

    def word(self, word):
        if len(word) > self.max_factors:
            self.stop('factor_limit', 'Rank-two elementary/diagonal word budget exhausted')
        self.check(*(item[-1] for item in word))

    @contextmanager
    def fast_probe(self):
        """Reserve a bounded prefix without resetting or estimating its usage."""
        self.check()
        self._probe = (self.steps+min(50000, max(1, (self.max_steps-self.steps)//8)),
                       self.products+min(1000000, max(1, (self.max_products-self.products)//8)),
                       min(self.deadline, perf_counter()+min(3.0, (self.deadline-perf_counter())/4)))
        try:
            yield
        finally:
            self._probe = None
