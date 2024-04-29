from sklearn.mixture import GaussianMixture

class GMM_Model:
    def __init__(self, n_components=5, covariance_type='full', random_state=42):
        """
        Initialize the Gaussian Mixture Model.
        
        Parameters:
        n_components (int): The number of mixture components.
        covariance_type (str): Type of covariance parameters to use. Options include 'full', 'tied', 'diag', 'spherical'.
        random_state (int): The seed used by the random number generator.
        """
        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)

    def fit(self, X):
        """
        Fit the GMM to the data using the Expectation-Maximization algorithm.
        
        Parameters:
        X (array-like of shape (n_samples, n_features)): Training data.
        """
        self.gmm.fit(X)

    def predict(self, X):
        """
        Predict the labels for the data samples in X using the model.
        
        Parameters:
        X (array-like of shape (n_samples, n_features)): Data to predict.
        
        Returns:
        array, shape (n_samples,): Component labels.
        """
        return self.gmm.predict(X)

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.
        
        Parameters:
        X (array-like of shape (n_samples, n_features)): Data to predict.
        
        Returns:
        array, shape (n_samples, n_components): The probability of the sample for each Gaussian component in the model.
        """
        return self.gmm.predict_proba(X)

    def score_samples(self, X):
        """
        Compute the weighted log probabilities for each sample.
        
        Parameters:
        X (array-like of shape (n_samples, n_features)): Data to score.
        
        Returns:
        array, shape (n_samples,): Log likelihood of the Gaussian mixture given the data.
        """
        return self.gmm.score_samples(X)
