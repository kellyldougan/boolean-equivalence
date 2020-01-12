import numpy as np
import itertools


class BooleanFunc:
    """
    Class which contains basic info about a given boolean function

    Args
    ----
    n : int
            degree of the function
    fn : list of array-like objects or array
        terms of the boolean function or truth table
    typ : str {'anf', 'tt'}
        type of function input
        """
    def __init__(self, n, fn, typ='anf'):
        self.n = n
        self.Vn = Bool(n).Vn
        if typ is 'anf':
            self.fn = fn
            self.fn_tt = self.tt
        elif typ is 'tt':
            if len(fn) != 2**n:
                raise ValueError('truth table needs to be size %d' % 2**n)
            self.fn_tt = fn
            self.fn = self.anf()
        else:
            raise Exception('typ is not understood')

    def anf(self):
        # list of coefficients
        C = []
        for i in range(self.Vn.shape[0]):
            c = 0
            for j in range(i+1):
                for k in range(self.n):
                    truth = 0
                    if self.Vn[i][k] < self.Vn[j][k]:
                        truth = 1
                        break
                    else:
                        continue
                if truth == 0:
                    c += self.fn_tt[j]
            C.append(c % 2)
        anf_array = self.Vn[np.array(C, dtype=bool)]
        anf_tups = []
        for term in anf_array:
            func_term = [i for i, tem in enumerate(term) if tem == 1]
            anf_tups.append(tuple(func_term))
        return anf_tups

    @staticmethod
    def _evalf(tup, x):
        """evaluate function tup (given by tuples) at point x"""
        f = 0
        for term in tup:
            mono = 1
            for i in term:
                mono *= x[i]
            f += mono
        return f % 2

    @property
    def tt(self):
        """Truth table of a given function"""
        L = []
        for term in self.Vn:
            L.append(self._evalf(self.fn, term))
        return np.array(L)

    def weight(self, start=None, end=None):
        """Weight of entire or portion of truth table
        (for parallel purposes)"""
        return sum(self.fn_tt[start:end])

    @property
    def walsh(self):
        """Returns the list of walsh transform values"""
        Walsh = []
        for w in self.Vn:
            w = np.array([int(i) for i in w])
            coeff = 0
            for i in range(2**self.n):
                x = np.array([int(k) for k in self.Vn[i]])
                coeff += (-1)**(self.fn_tt[i] + np.dot(w, x))
            Walsh.append(coeff)
        return Walsh

    @property
    def nonlinearity(self):
        return (1/2)*(2**self.n-max([abs(i) for i in self.walsh]))

    @property
    def autocorr(self):
        """Returns the list of autocorrelation values"""
        V = self.Vn
        r = []
        for i in range(2**self.n):
            d = V[i]
            ac = 0
            for j in range(2**self.n):
                x = V[j]
                xdsum = ''
                for k in range(len(x)):
                    xdsum += str((int(x[k]) + int(d[k])) % 2)
                ac += (-1)**(self.fn_tt[j] + self.fn_tt[int(xdsum, 2)])
            r.append(ac)
        return r

    @property
    def abs_ind(self):
        return max([abs(i) for i in self.autocorr[1:]])


class Bool:
    """
    Class which contains basic info about boolean functions of degree n

    Args
    ----
    n : int
        degree
    """
    def __init__(self, n):
        self.n = n

    @property
    def Vn(self):    # Creates Vn, boolean vector space
        return np.array(list(itertools.product(*[(0, 1)] * self.n)))

    def ind(self, indicator):
        """Use this if you don't want to use a rotation symmetric function
        Example
        -------
        if f(x) = x0x1+x1x3, do
        Ind([[0,1],[1,3]]) = [(0,1), (1,3)]
        """
        b = []
        for c in indicator:
            c = sorted(c)
            b.append(tuple(c))
        fn = tuple(sorted(b))
        return BooleanFunc(self.n, fn)

    def mrs(self, term, k=1):
        """Return all tuples of the MRS function given by a term

        Keyword Args
        ------------
        k : int
            integer which specifies amount to increment each term

        Example
        -------
        myfunc = Bool(4)
        myfunc.MRS((0,1)) = [(0,1),(1,2),(2,3),(0,3)]
        myfunc.MRS((0,2), k=2) = [(0,2)]
        """

        b = []
        a = sorted(term)
        b.append(tuple(a))
        for i in range(0, self.n-1):
            a = np.array(a)
            a = (a + k) % self.n
            a = list(a)
            b.append(tuple(sorted(a)))
        fn = tuple(set(b))
        return BooleanFunc(self.n, fn)


class PermEquivalence(Bool):
    """class to check if two functions are equivalent by a permutation"""
    def __init__(self, n, f1=None, f2=None, parity=True):
        """
        Args
        ----
        n : int
            degree of the functions

        Keyword Args
        ------------
        f1, f2 : array-like
            Truth tables of the two functions
        parity : bool
            If False, checks all permutations
            If True, checks only permutations which preserve parity
            Default is True
        """
        super().__init__(n)
        if not parity:
            self.allperms = np.array(list(itertools.permutations(range(n))))
        else:
            self.allperms = self._parity_perms
        self.f1 = np.asarray(f1)
        self.f2 = np.asarray(f2)

    @property
    def _parity_perms(self):
        """create all permutations which send evens to evens, odds to odds
        or evens to odds, odds to evens"""
        evens = np.array(list(itertools.permutations(range(0, self.n, 2))))
        odds = np.array(list(itertools.permutations(range(1, self.n, 2))))
        num_evens = evens.shape[0]
        num_odds = evens.shape[0]
        allperms = np.empty((2*num_evens*num_odds, self.n))
        for i, even in enumerate(evens):
            allperms[i*num_odds: (i + 1)*num_odds, [range(0, self.n, 2)]] = \
                even
            for j, odd in enumerate(odds):
                allperms[j+i*num_odds, [range(1, self.n, 2)]] = odd
        for i, even in enumerate(evens):
            allperms[(i + num_evens)*num_odds:(i+1+num_evens)*num_odds,
                     [range(1, self.n, 2)]] = even
            for j, odd in enumerate(odds):
                allperms[j+(i+num_evens)*num_odds, [range(0, self.n, 2)]] = odd
        return np.array(allperms, dtype=int)

    def perm_to_matrix(self, perm):
        """convert a permutation to a matrix"""
        n = self.n
        A = np.zeros((n, n))
        for i in range(n):
            A[i, perm[i]] = 1
        return A

    def perm_equiv(self):
        """Determine whether two functions are equivalent by permutation"""
        TT1 = self.f1
        TT2 = self.f2
        for i in range(len(self.allperms)):
            val = True
            j = 0
            perm = self.allperms[i]
            # go through and see if f(perm(x))=g(x) for all x in Vn
            while j in range(0, len(TT1)) and val is True:
                # permute a given vector of Vn
                check_entry = np.dot(self.perm_to_matrix(perm), self.Vn[j]) % 2
                # convert the dot product into an int
                ttentry = int(''.join(str(int(k)) for k in
                                      list(check_entry)), 2)
                if TT1[ttentry] != TT2[j]:
                    val = False
                j += 1
            if val is True:
                print('Permutation:', perm+1)
                return True
        return False

if __name__ == '__main__':
    import math
    f1 = [1,0,1,1,1,0,0,1]
    n = 3
    f = BooleanFunc(3,f1, typ='tt')
    print(f.anf())
