import random
import unittest
from css_decoupling.two_d.algebra import Ring,Matrix,eye,inverse,ResourceLimit,Lattice,FiniteQuotient
from css_decoupling.two_d.models import standard_code
from css_decoupling.two_d.qca import invert_physical_chain,construct_qca


class ChainInverseTests(unittest.TestCase):
    def test_matches_known_and_generic_inverses(self):
        R=Ring(2,2);rng=random.Random(1051)
        for t,px,pz in ((0,0,0),(0,2,2),(1,0,0),(2,1,2)):
            X,Z=standard_code(t,product_x=px,product_z=pz)
            n=X.ncols;U,V=eye(R,n),eye(R,n)
            if n>1:
                for _ in range(5):
                    i,j=rng.sample(range(n),2);E=eye(R,n)
                    E[i,j]=R.monomial((rng.randrange(-2,3),rng.randrange(-2,3)))
                    U=E*U;V=V*E
            if n:
                shift=eye(R,n);restore=eye(R,n)
                shift[0,0]=R.monomial((-2,1));restore[0,0]=R.monomial((2,-1))
                U=shift*U;V=V*restore
            HX=X*U;HZ=(V*Z.dagger()).dagger()
            actual=invert_physical_chain(V,HX,HZ,X,Z,eye(R,X.nrows),eye(R,Z.nrows),px,pz,t)
            self.assertEqual(actual,U);self.assertEqual(actual,inverse(V))

    def test_bad_endpoint_and_budget_rejected(self):
        X,Z=standard_code(1,product_z=1);R=X.ring;V=eye(R,3)
        with self.assertRaises((ValueError,ArithmeticError)):
            invert_physical_chain(V,X,Z,X,Z,eye(R,1),Matrix(R,2,2),0,1,1)
        with self.assertRaises(ResourceLimit):
            invert_physical_chain(V,X,Z,X,Z,eye(R,1),eye(R,2),0,1,1,max_expansion_terms=0)

    def test_restriction_context_checked_before_positive_shortcut(self):
        X,Z=standard_code(1);L=Lattice([[1,0],[0,1]])
        for cap in (0,True,1.5):
            with self.subTest(cap=cap),self.assertRaises(ValueError):
                construct_qca(X,Z,1,restriction_context={'HX':X,'HZ':Z,'lattice':L,'max_terms':cap})
        Q=FiniteQuotient(X);Q.actions=(Matrix(X.ring,1,1),Q.actions[1])
        with self.assertRaises(ValueError):
            construct_qca(X,Z,1,restriction_context={'HX':X,'HZ':Z,'lattice':L,'quotient':Q})


if __name__=='__main__':unittest.main()
