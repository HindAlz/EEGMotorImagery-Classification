from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CSPLDA:
    def __init__(
        self,
        n_components=8,
        reg=None,
        log=True,
        norm_trace=False,
    ):
        self.csp = CSP(
            n_components=n_components,
            reg=reg,
            log=log,
            norm_trace=norm_trace,
        )
        self.lda = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        features = self.csp.fit_transform(X, y)
        self.lda.fit(features, y)
        return self

    def predict(self, X):
        features = self.csp.transform(X)
        return self.lda.predict(features)

    def predict_proba(self, X):
        features = self.csp.transform(X)
        return self.lda.predict_proba(features)