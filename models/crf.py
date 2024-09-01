from sklearn_crfsuite import CRF

from .util import sent2features


class CRFModel(object):
    """
    条件随机场通过引入自定义的特征函数，不仅可以表达观测之间的依赖，
    还可表示当前观测与前后多个状态之间的复杂依赖，可以有效克服HMM模型面临的问题。
    """
    def __init__(self,
                 algorithm='lbfgs',
                 c1=0.1,#L1 regularization(L1范数正则化)的系数
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False #generate (L * L) transition features
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        """
        Train a model.
         sentences : list of lists of dicts
            Feature dicts for several documents (in a python-crfsuite format).

        tag_lists : list of lists of strings
            Labels for several documents.
        """
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        """
        Make a prediction.

        Parameters
        ----------
        sentences : list of lists of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of lists of strings
            predicted labels

        """
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists

    def set(self,algorithm,c1,c2,max_iterations,):
        self.model.algorithm=algorithm
        self.model.c1=c1
        self.model.c2=c2
        self.model.max_iterations=max_iterations
