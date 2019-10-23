from mlxtend.frequent_patterns import apriori
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

def cover(X, itemset): # could be called "rows_covered_by_itemset"
    """
    Returns the rows in X which satisfy the itemset
    An itemset is satisfied when all elements of the itemset are evaluated to 1 (True)

    Input:
        X: pandas DataFrame (all one-hot encoded)
        itemset: an iterable of column names representing an itemset

    Returns:
        X (subset): pandas DataFrame whose rows satisfy all itemset elements (have value 1)
    """
    return X[(X[itemset]==1).all(axis=1)]

def overlap(X, itemset1, itemset2): # could be called "rows_covered_by_intersection_of_itemsets"
    """
    Returns the rows in X which satisfy BOTH itemset1 and itemset2 (their intersection)
    An itemset is satisfied when all elements of the itemset are evaluated to 1 (True)

    Input:
        X: pandas DataFrame (all one-hot encoded)
        itemset1: an iterable of column names representing an itemset
        itemset2: an iterable of column names representing an itemset

    Returns:
        X (subset): pandas DataFrame whose rows satisfy all itemset elements (have value 1)
    """
    cover1 = cover(X, itemset1)
    cover2 = cover(X, itemset2)
    overlap_idx = cover1.index.intersection(cover2.index)
    return X.loc[overlap_idx]

def correct_cover(X, y, rule):
    """
    Returns elements of X (and y) which satisfy the rule (item, class)

    Input:
        X: pandas DataFrame (all one-hot encoded) of input data
        y: pandas Series of categorical target variable (ground truth)
        rule: element of pandas DataFrame with `item` (representing a rule)
              and `class` (representing a prediction)

    Returns:
        X_cover, y_cover: pandas DataFrame and Series that are correctly covered by the rule
    """
    item = rule['item']
    pred = rule['class']

    X_positive = X.loc[y == pred]      # rows where y == class
    X_cover = cover(X_positive, item)  # rows covered by the rule
    y_cover = y[X_cover.index]
    return X_cover, y_cover

def incorrect_cover(X, y, rule):
    """
    Returns the incorrect cover, defined as the set difference of cover(r) \ correct_cover(r)
    Where `r` is the rule

    Input:
        X: pandas DataFrame (all one-hot encoded) of input data
        y: pandas Series of categorical target variable (ground truth)
        rule: element of pandas DataFrame with `item` (representing a rule)
              and `class` (representing a prediction)

    Returns:
        X_incorrect, y_incorrect: pandas DataFrame and Series that are incorrectly covered by rule
    """
    item = rule['item']
    pred = rule['class']

    X_negative = X.loc[y != pred]      # rows where y == class
    X_cover = cover(X_negative, item)  # rows covered by the rule
    y_cover = y[X_cover.index]
    return X_cover, y_cover

class DecisionSet():
    """
    Implementation of Decision Sets
    See here: https://www-cs-faculty.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf
    """
    def __init__(self, min_support = 0.01, lambdas = [1, 1, 1, 1, 100, 1, 100]):
        """
        Initializes a DecisionSet

        Input:
            min_support: float in (0, 1) denoting minimum proportion of rows an itemset can cover
            lambdas: len=7 iterable; values to use for weighting the loss function (lambdas in paper)
        """
        self.min_support = min_support
        self.lambdas = lambdas
        assert len(self.lambdas) == 7, '`lambdas` must be exactly length 7!'

    def fit(self, X, y, d1=1/3, d2=1/3, d3=-1):
        """
        Fits a DecisionSet to a dataset X (one-hot encoded) with labels y (categorical)

        Input:
            X: pandas DataFrame containing one-hot encoded (True/False or 1/0) values.
               X should have shape (n, p) where n is number of points and p is number of features
            y: pandas Series containing class labels
            d*: delta parameters. d1 : delta, d2/d3 : delta' (two settings)
        """
        itemsets = self.mine_itemsets(X)
        self.L_max = np.max(itemsets.apply(lambda x: len(x)))
        self.default_pred = y.value_counts().idxmax()

        # two runs each with different d' (d2 vs d3)
        dset1, obj_val1, dset2, obj_val2 = self.smooth_local_search(X, y, itemsets, d1=d1, d2=d2, d3=d3)

        # take best of both runs
        if obj_val1 > obj_val2:
            self.decision_set = dset1
        else:
            self.decision_set = dset2

        # sort by score for tie-breaking (and general presentation)
        print('Calculating individual rule scores')
        # f1_list = []
        acc_list = []
        for i, r in self.decision_set.iterrows():
            xcov = cover(X, r['item'])
            ycov = y.loc[xcov.index]
            pred = r['class'] * np.ones_like(ycov.values)
            if len(xcov) > 0:
                # f1 = f1_score(ycov, pred, labels=np.unique(pred))
                acc = accuracy_score(ycov, pred)
                # f1_list.append(f1)
                acc_list.append(acc)
        # self.decision_set['f1'] = f1_list
        self.decision_set['acc'] = acc_list
        self.decision_set = self.decision_set.sort_values(by='acc', ascending=False) # prioritize accuracy

        self._remove_duplicate_itemsets() # postprocess for visual appeal (does not affect accuracy or model performance)

        return self

    def predict(self, X):
        """
        make predictions for every element of X (pandas DataFrame)
        uses highest-F1 rule that applies, otherwise uses default
        """
        try:
            ds = self.decision_set
        except:
            raise RuntimeError('Must fit decision set before predicting!')

        preds = []
        for i, x in X.iterrows():
            added = False
            for j, r in self.decision_set.iterrows(): # sorted by some notion of how good the rule is (breaks ties)
                rule_value = x.loc[r['item']]
                if (rule_value==1).all():
                    preds.append(r['class'])
                    added = True
                    break
            if not added:
                preds.append(self.default_pred) # if no match, use default
        return np.array(preds)

    def smooth_local_search(self, X, y, itemsets, d1=1/3, d2=1/3, d3=-1):
        """
        Performs smooth local search to optimize decision set
        """
        domain = self._get_domain(itemsets, y)
        print('Rules in domain = {}'.format(len(domain)))

        print('Preprocessing...')
        # cache the covers (just cache the length)
        self.cover_cache = {}
        for i, r in itemsets.iteritems():
            self.cover_cache[r] = len(cover(X, r))

        # cache the overlaps (just cache the length)
        self.overlap_cache = {}
        for i, ri in itemsets.iteritems():
            for j, rj in itemsets.loc[i:].iteritems():
                self.overlap_cache[(ri,rj)] = len(overlap(X, ri, rj))

        # cache the correct covers (cache the actual set)
        self.correct_cover_cache = {}
        for i, r in domain.iterrows():
            # print(len(correct_cover(X, y, r)[0]))
            self.correct_cover_cache[(r['item'], r['class'])] = correct_cover(X, y, r)

        # cache incorrect cover (just cache the length)
        self.incorrect_cover_cache = {}
        for i, r in domain.iterrows():
            self.incorrect_cover_cache[(r['item'], r['class'])] = len(incorrect_cover(X, y, r)[0])
        print('Done')

        # initialize an empty decision set
        A = pd.DataFrame([], columns=['item', 'class'])
        # sample a decision set with no bias
        samp = self._sample_decision_set(domain, domain, 0) # line 4 of SLS algorithm
        # estimate the objective with the sample
        opt = self._objective(X, y, itemsets, samp)
        self.OPT = opt
        self.error_bound = opt / (len(domain) ** 2)
        print('Error bound = {}'.format(self.error_bound))

        while True:
            print('Estimating w...')
            w = self._estimate_w(domain, A, X, y, itemsets, d1=d1)
            print('Done')
            # print('w = {}'.format(w))

            print('Adding elements to A ')
            A, add_done = self._add_elements(domain, A, w)
            print('Current A:\n{}'.format(A))
            if not add_done:
                continue # go back to top of loop
            else:
                print('Removing elements from A')
                A, remove_done = self._remove_elements(domain, A, w)
                print('Current A:\n{}'.format(A))
                if not remove_done:
                    continue
                else:
                    break

        # we run with d' = 1/3 and d' = -1 according to the paper (Section 4.2)
        dset1 = self._sample_decision_set(domain, A, d2)
        obj1 = self._objective(X, y, itemsets, dset1)
        dset2 = self._sample_decision_set(domain, A, d3)
        obj2 = self._objective(X, y, itemsets, dset2)
        return dset1, obj1, dset2, obj2

    def mine_itemsets(self, X):
        """
        Performs frequent itemset mining using Apriori Algorithm
        See here: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

        Input:
            X: pandas DataFrame, one-hot encoded

        Output:
            itemsets: pandas DataFrame with support and itemsets columns
        """
        itemsets = apriori(X, min_support=self.min_support, use_colnames=True)
        return itemsets.itemsets # apriori returns the support for each item. just take itemsets

    def _get_domain(self, itemsets, y):
        """
        calculate cartesian product of itemset and classes, returns a dataframe
        """
        self.C = y.unique()
        item_cross_class = itertools.product(itemsets, self.C)
        domain = pd.DataFrame(item_cross_class, columns=['item', 'class'])
        return domain

    def _add_elements(self, domain, A, w):
        """
        implements lines 9-12 of algorithm 1
        """
        pre_shape = A.shape
        added = 0
        for i in range(len(domain)):
            r = domain.iloc[i]
            in_A = (A == r).all(axis=1).any()
            if not in_A and w[i] > 2*self.error_bound:
                A = A.append(r)
                added += 1
        print('Executed _add_elements, shape change of A: {} -> {}'.format(pre_shape, A.shape))
        if added == 0:
            return A, True
        else:
            return A, False

    def _remove_elements(self, domain, A, w):
        """
        implements lines 13-15 of algorithm 1
        """
        pre_shape = A.shape
        removed = 0
        for i in range(len(domain)):
            r = domain.iloc[i]
            in_A = (A == r).all(axis=1).any()
            if in_A and w[i] < -2*self.error_bound:
                to_drop = A.loc[(A == r).all(axis=1)]
                print('dropping: {}'.format(to_drop))
                A = A.drop(to_drop.index)
                removed += 1
        print('Executed _remove_elements, shape change of A: {} -> {}'.format(pre_shape, A.shape))
        if removed == 0:
            return A, True
        else:
            return A, False

    def _sample_decision_set(self, domain, decision_set, delta):
        """
        performs sampling of decision set according to algorithm in paper (Definition 8)
        samples with bias (if already in decision_set) from domain
        """
        # bias towards within decision_set / domain, depending on delta
        p_in = (1 + delta) / 2
        p_out = (1 - delta) / 2

        # sample from decision set w.p. p_in
        R = decision_set.shape[0]
        in_mask = np.random.random(size=R) < p_in
        in_samp = decision_set[in_mask]

        # sample from domain \ decision_set (set difference) w.p. p_out
        out_domain = domain.loc[domain.index.difference(decision_set.index)] # domain \ decision_set
        S = out_domain.shape[0]
        out_mask = np.random.random(size=S) < p_out
        out_samp = out_domain[out_mask]

        sample = pd.concat([in_samp, out_samp])
        return sample

    def _objective(self, X, y, itemsets, decision_set):
        """
        Computes the objective outlined in the Decision Set Paper
        """
        N = len(X)
        R = len(decision_set)
        S = len(itemsets)

        # interpretability: less rules (possible for R > S ?)
        f1 = S - R

        # interpretability: short rules
        f2 = self.L_max * S - sum([len(r['item']) for i, r in decision_set.iterrows()])

        # non-overlap: discourage overlap among rules
        f3, f4 = self._non_overlap_objective(X, itemsets, decision_set)

        # coverage: at least one rule for each class
        f5 = self._coverage_objective(decision_set)

        # accuracy: reduce incorrect-cover (precision)
        f6 = self._precision_objective(X, y, decision_set, itemsets)

        # accuracy: cover data points with at least one rule (recall)
        f7 = self._recall_objective(X, y, decision_set)

        objectives = [f1, f2, f3, f4, f5, f6, f7]
        # final objective: weighted sum of individuals
        return sum([o * l for o, l in zip(objectives, self.lambdas)])

    def _non_overlap_objective(self, X, itemsets, decision_set):
        """
        perform nested for-loop summation for calculating the non-overlap objective
        """
        N = len(X)
        R = len(decision_set)
        S = len(itemsets)

        # calculate overlap for same and different classes separately
        same_class_overlap, diff_class_overlap = 0, 0

        for (i, (idx_i, ri)) in enumerate(decision_set.iterrows()):
            for (j, (idx_j, rj)) in enumerate(decision_set.iloc[i:].iterrows()):
                try:
                    num_overlap = self.overlap_cache[(ri['item'], rj['item'])]
                except:
                    num_overlap = self.overlap_cache[(rj['item'], ri['item'])]
                if ri['class'] == rj['class']:
                    same_class_overlap += num_overlap
                else:
                    diff_class_overlap += num_overlap

        # same-class objective ('f3' in paper)
        f3 = N * (S ** 2) - same_class_overlap
        # diff-class objective ('f4' in paper)
        f4 = N * (S ** 2) - diff_class_overlap

        return f3, f4

    def _coverage_objective(self, decision_set):
        """
        calculates coverage objective (f5 in paper)
        """
        obj = 0
        for cls in self.C:
            have_rule = (decision_set['class']==cls).any(axis=0)
            obj += 1 if have_rule else 0
        return obj

    def _precision_objective(self, X, y, decision_set, itemsets):
        """
        calculates precision objective (f6 in paper)
        """
        N = len(X)
        S = len(itemsets)
        R = len(decision_set)

        penalty = 0
        for i, r in decision_set.iterrows():
            penalty += self.incorrect_cover_cache[(r['item'], r['class'])]

        return N * (S ** 2) - penalty

    def _recall_objective(self, X, y, decision_set):
        """
        calculates recall objective (f7 in paper)
        """
        xcov, ycov = correct_cover(X, y, decision_set.iloc[0])
        idx = xcov.index
        for (j, r) in decision_set.iloc[1:].iterrows():
            xcov, ycov = self.correct_cover_cache[(r['item'], r['class'])]
            idx = idx.union(xcov.index)
        return len(idx)

    def _estimate_w(self, domain, A, X, y, itemsets, d1=1/3, maxiter=50):
        """
        performs the w estimation from line 5 of algorithm
        """
        w = []
        for i, r in domain.iterrows():
            std_error = float('inf')
            diffs = []
            j = 0
            while std_error > self.error_bound and j < maxiter:

                d_samp1 = self._sample_decision_set(domain, A, d1)
                d_samp_with = d_samp1.append(r).drop_duplicates(keep='first')
                d_samp_drop = d_samp_with.append(r).drop_duplicates(keep=False)

                obj_with = self._objective(X, y, itemsets, d_samp_with)
                obj_drop = self._objective(X, y, itemsets, d_samp_drop)

                diffs.append(obj_with - obj_drop)

                if len(diffs) >= 2:
                    std_error = np.std(diffs)
                j += 1

            w.append(np.mean(diffs))
        return w

    def _remove_duplicate_itemsets(self):
        """
        removes any duplicate itemsets (i.e. itemsets which predict different classes)
        keeps the first entry, which is assumed to have already been sorted by some notion of score
        i.e. keeps the "best"
        """
        self.decision_set = self.decision_set.loc[self.decision_set['item'].drop_duplicates(keep='first').index]
