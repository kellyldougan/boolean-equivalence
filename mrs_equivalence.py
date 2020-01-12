import numpy as np
import itertools
from math import gcd


class CubicEquivalence:
    def __init__(self, n):
        self.n = int(n)

    @staticmethod
    def _mu(x, sig2):
        return (np.array(x)-1)*(sig2-1) + 1

    @property
    def funcs(self):
        allfuncs = [(1, a, b) for a in range(2, int(self.n/3)+2)
                    for b in range(3, self.n)]
        uniquefuncs = [x for x in allfuncs if
                       (2*x[1]-1 <= x[2] and x[2] <= self.n - x[1] + 1) or
                       (self.n % 3 == 0 and x[1] == int(self.n / 3) + 1 and
                        x[2] == 2 * int(self.n/3)+1)]
        return uniquefuncs

    def one_terms(self, term):
        """Returns all one terms of a cubic function"""
        a, b, c = sorted(list(term))
        term = np.array(term) - a + 1
        term1 = tuple(sorted(term))
        term2 = (1, term1[2]+1-term1[1], self.n + 2 - term1[1])
        term3 = (1, self.n+2-term1[2], self.n + term1[1]+1-term1[2])
        return list(set([term1, term2, term3]))

    def _freduce(self, term):
        term1 = tuple(sorted(term))
        term2 = (1, term1[2]+1-term1[1], self.n + 2 - term1[1])
        term3 = (1, self.n+2-term1[2], self.n + term1[1]+1-term1[2])
        return min([term1, term2, term3])

    @property
    def equiv_classes(self):
        fun = self.funcs
        equivclasses = {fun[i]: i for i in range(len(fun))}
        for sig2 in range(3, self.n+1):
            if gcd(sig2-1, self.n) == 1:
                for i in range(len(fun)):
                    newfun = self._freduce((self._mu(fun[i],
                                                     sig2)-1)
                                           % self.n+1)
                    newval = min(equivclasses[fun[i]], equivclasses[newfun])
                    equivclasses[fun[i]] = equivclasses[newfun] = newval
        inds = set(equivclasses.values())
        classes = [[f for f in equivclasses.keys() if equivclasses[f] == i]
                   for i in inds]
        ec_dict = {}
        for ec in classes:
            ec = sorted(ec)
            rep = ec[0]
            size = len(ec)
            ec_dict[rep] = {'size': size, 'elts': ec}
        return ec_dict  
    
    @property
    def num_classes(self):
        return len(self.equiv_classes)
    

class QuarticEquivalence(CubicEquivalence):
    def __init__(self, n):
        if n % 2 == 1:
            raise ValueError('n must be even')
        super().__init__(n)
        self.m = int(n/2)

    @staticmethod
    def _mod(term, n):
        """Return the term mod n 
        Example
        -------
        self._mod([1,2,14,8],8)=[1,2,6,8]"""
        n_terms = [a for a in term if a % n == 0]
        if len(n_terms) == 0:
            return term % n
        else:
            reduced = term % n
            zero_ind = np.argmin(reduced)
            reduced[zero_ind] = n
            return reduced

    def _lexico_sort(self, term):
        """Return the lexicographically least term of the function"""
        odds = [a for a in term if a % 2 == 1]
        if len(odds) == 1:
            return np.array(sorted(self._mod(term, self.n)))
        elif len(odds) == 0:
            two_terms = []
            for i in term:
                equiv_term = term+2-i
                equiv_term = self._mod(equiv_term, self.n)
                two_terms.append(sorted(equiv_term))
            return np.array(min(two_terms))
        elif len(odds) > 1:
            one_terms = []
            for i in odds:
                equiv_term = term+1-i
                equiv_term = self._mod(equiv_term, self.n)
                one_terms.append(sorted(equiv_term))
            return np.array(min(one_terms))


class Mf1Equivalence(QuarticEquivalence):
    def __init__(self, n):
        super().__init__(n)

    @property
    def funcs(self):
        """Create lexico-least one terms of all mf1 functions"""
        cubic_evens = np.array(list(itertools.combinations(range(2,
                                                                 self.n+1,
                                                                 2),
                                                           3)))
        evens = np.insert(cubic_evens, 0, 1, axis=1)
        cubic_odds = np.array(list(itertools.combinations(range(1, self.n, 2),
                                                          3)))
        odds = np.insert(cubic_odds, 0, 2, axis=1)
        all_terms = np.vstack((evens, odds))
        sorted_terms = sorted([list(self._lexico_sort(term))
                               for term in all_terms])
        return np.array(sorted_terms)

    def pattern(self, term):
        """Find pattern of cubic part of term in mf1 function"""
        even = [i for i in term if i % 2 == 0]
        odd = [i for i in term if i % 2 == 1]
        if len(even) == 3 and len(odd) == 1:
            term = even
        elif len(even) == 1 and len(odd) == 3:
            term = odd
        else:
            raise TypeError('term should be mf1')
        i, j, k = term
        return [(j-i) % self.n, (k-i) % self.n, (k-j) % self.n]

    @property
    def _cubic_to_quartic(self):
        """Change a list of cubic functions
        to a list of quartic mf1 functions"""
        funcs = self.funcs
        to_even = np.sum(funcs % 2, axis=1) == 1
        to_odd = np.sum(funcs % 2, axis=1) == 3
        cubic = np.zeros((len(funcs), 3), dtype=int)
        cubic[to_even] = funcs[to_even][funcs[to_even]%2 == 0].reshape((-1,3))/2
        cubic[to_odd] = (funcs[to_odd][funcs[to_odd]%2 == 1].reshape((-1,3))+1)/2
        reduced_ec = CubicEquivalence(self.m)
        cubic = np.vstack([reduced_ec._freduce(row) for row in cubic])
        unique_cubs = {tuple(row) for row in cubic}
        cubic_lookup = {cub: funcs[np.all(cubic == cub, axis=1)]
                        for cub in unique_cubs}
        return cubic_lookup

    @property
    def equiv_classes(self):
        """Return dictionary with representative term as the keys,
        a dictionary with size and and all terms in class as values"""
        cubic_classes = CubicEquivalence(self.m).equiv_classes
        quartic_classes = {}
        funcs = self._cubic_to_quartic
        for info in cubic_classes.values():
            cub_ec = info['elts']
            elts = np.sort(np.vstack([funcs[cub] for cub in cub_ec]), axis=0)
            size = len(funcs)
            quartic_classes[tuple(elts[0])] = {'size': size, 'elts': elts}
        return quartic_classes

    @property
    def num_classes(self):
        return len(self.equiv_classes)


class Mf2Equivalence(QuarticEquivalence):

    def __init__(self, n):
        super().__init__(n)
        self.chi_pairs = np.array(sorted([(a, b)
                                  for a in range(2, self.m+1, 2)
                                  for b in range(a, self.m+1, 2)]))
        self.U_m = [k for k in range(self.m) if gcd(k, self.m) == 1]

    def term_to_chi(self, term):
        term = np.asarray(term)
        if len(term) != 4:
            raise ValueError('term needs to be quartic')
        if sum(term % 2) != 2:
            raise ValueError('term needs to be mf2')
        evens = np.array(sorted([i for i in term if i % 2 == 0]))
        odds = np.array(sorted([i for i in term if i % 2 == 1]))
        chi1 = np.diff(evens)
        chi2 = np.diff(odds)
        return sorted([chi1[0], chi2[0]])

    def is_equiv(self, chi1, chi2):
        """ Returns true if mrs funcs with chi values chi1, chi2
            are equivalent.
        Args
        ----
        chi1: array-like
            first pair of chi values
        chi2: array-like
            second pair of chi values
        """
        chi1 = np.asarray(chi1)
        chi2 = np.asarray(chi2)
        if sum(chi1 % 2) != 0 or sum(chi2 % 2) != 0:
            raise ValueError('chi pairs should be even valued')
        #testing
        gcd_list = self.U_m
        for k in gcd_list:
            chi1_k = (k*chi1) % self.n
            chi1_rep = sorted([min(chi1_k[i], self.n-chi1_k[i])
                               for i in range(2)])
            if chi1_rep == list(chi2):
                return True
        return False

    def chi_rep(self, chi_pair):
        """Returns one term represented by chi pair.

        Args
        ----
        chi_pair: array-like
            pair of chi values.
        """
        return [1, 2, (1+chi_pair[0]) % self.n, (2+chi_pair[1]) % self.n]

    @property
    def chi_classes(self):
        """Returns dictionary with keys being a equivalence class
        representative term, values a dictionary of the chi-pair
        representatives, term representative, and size."""
        classes = {}
        # list of lists of equivalence classes with chi-pair reps
        chi_reps = []
        for chi in self.chi_pairs:
            found = False
            for c in chi_reps:
                if self.is_equiv(c[0], chi):
                    c.append(chi)
                    found = True
                    break
            if not found:
                chi_reps.append([chi])
        for ec in chi_reps:
            size = len(ec)
            term_reps = []
            for rep in ec:
                term_reps.append(self.chi_rep(rep))
            rep = tuple(min(term_reps))
            ec_info = {'size': size, 'chi values': ec, 'elts': term_reps}
            classes[rep] = ec_info
        return classes

    @property
    def reps(self):
        return list(self.chi_classes.keys())

    @property
    def num_classes(self):
        return len(self.chi_classes.keys())

    @property
    def mates(self):
        """Separate self-mate reduced chis (pairs reduced by 2)"""
        chis = np.array([self.term_to_chi(rep) for rep in self.reps])
        reduced = np.array([chi/2 for chi in chis], dtype=int)
        U_m = self.U_m
        mates = {}
        mated = []
        self_mates = []
        for pair in reduced:
            a, b = pair
            is_mate = False
            for k in U_m:
                new = sorted((k*pair) % self.m)
                mate = sorted(np.array([a, -b]) % self.m)
                if np.all(new == mate):
                    self_mates.append(pair)
                    is_mate = True
                    break
            if is_mate is False:
                mated.append(pair)
        mates['self_mate'] = np.array(self_mates)
        mates['mated'] = np.array(mated)
        return mates

if __name__ == '__main__':
    n = 30
    mf2 = Mf2Equivalence(n)
    print(mf2.mates)

