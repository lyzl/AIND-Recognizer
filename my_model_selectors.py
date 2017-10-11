import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #L: likelihood of the fitted model
        #p: number of parameters,
        #N: number of data points.
        best_score = float("inf")
        best_num_components = None
        # print(self.X)
        # print(self.lengths)
        # print(self.sequences)
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                # score has already logged
                logL = hmm_model.score(self.X, self.lengths)
                logN = math.log(len(self.X))
                p = n * n + 2 * n * len(self.X[0]) - 1
                BIC_result = (-2) * logL - p * logN
                
                if BIC_result < best_score:
                    best_score = BIC_result
                    best_num_components = n
                if self.verbose:
                    print("BIC score:{}".format(BIC_result))
            except:
                if self.verbose:
                    print("failure in selector BIC.The error is:{}".format(sys.exc_info()[0]))
                pass
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    # from IPython.core.debugger import set_trace
    # set_trace()
    def select(self):
        # warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        score_list = []
        best_score = float("inf")
        best_num_components = None
        M = len(self.words)
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                log_prob_xi = hmm_model.score(self.X, self.lengths)
                
                # print(log_prob_xi)
                score_list.append(log_prob_xi) 
                # print(score_dict.items())
            except:
                score_list.append(None)
                if self.verbose:
                    print("failure in selector DIC.The error is:{}".format(sys.exc_info()))
                pass

        for n in range(0, self.max_n_components - self.min_n_components):
            if score_list[n] != None:
                log_prob_xi = score_list[n]
                log_prob_but_i = sum([score_list[m] for m in range(0, self.max_n_components - self.min_n_components + 1) if ((score_list[m] != None) and (m != n))])
                DIC_result = log_prob_xi - (1 / (M - 1) *log_prob_but_i)

                if DIC_result < best_score:
                    best_score = DIC_result
                    best_num_components = n + self.min_n_components
        if self.verbose:
            print("DIC score:{}".format(DIC_result))
        return self.base_model(best_num_components)
        # return None

        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("inf")
        best_num_components = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                score_list = []
                n_splits = min(3,len(self.sequences))
                KFold_split = KFold(n_splits = n_splits, random_state = self.random_state)
                for train_idx, test_idx in KFold_split.split(self.sequences):
                    train_X, train_length = combine_sequences(train_idx, self.sequences)
                    test_X, test_Length = combine_sequences(test_idx, self.sequences)
                    hmm_model.fit(train_X, train_length)
                    splited_score = hmm_model.score(test_X, test_Length)
                    score_list.append(splited_score)

                average_score = sum(score_list)/len(score_list)
                    
                if average_score < best_score :
                    best_score = average_score
                    best_num_components = n
            except:
                if self.verbose:
                    print("failure in selector CV.The error is:{}".format(sys.exc_info()[0]))
                pass
        if self.verbose:
            print("CV score:{}".format(DIC_result))
        return self.base_model(best_num_components)

                        
