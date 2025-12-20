"""
STM-style topic modeling utilities.

Wraps sklearn's NMF/LDA to provide an STM-like interface with:
- Document-topic proportions (theta)
- Topic-term distributions (beta)
- Top terms per topic
- Optional document covariates (temporal effects)

Note: Full STM (Roberts et al. 2014) is implemented in R. This is a 
Python approximation using standard topic models for the 2-month validation.
For full thesis, consider rpy2 integration with R's stm package.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
from pathlib import Path


@dataclass
class TopicModelResults:
    """
    Container for topic model outputs.
    
    Attributes
    ----------
    model : sklearn topic model
        Fitted NMF or LDA model
    vectorizer : sklearn vectorizer
        Fitted CountVectorizer or TfidfVectorizer
    theta : np.ndarray
        Document-topic proportions (n_docs × K)
        Each row sums to 1 (for LDA) or is normalized (for NMF)
    feature_names : list of str
        Vocabulary (ordered by vectorizer)
    n_topics : int
        Number of topics K
    method : str
        'nmf' or 'lda'
    random_state : int
        Random seed used
    """
    model: object
    vectorizer: object
    theta: np.ndarray
    feature_names: List[str]
    n_topics: int
    method: str
    random_state: int
    
    def get_top_terms(self, topic_id: int, n_terms: int = 15) -> List[Tuple[str, float]]:
        """
        Get top terms for a topic.
        
        Parameters
        ----------
        topic_id : int
            Topic index (0-based)
        n_terms : int
            Number of top terms to return
            
        Returns
        -------
        list of (str, float)
            [(term, weight), ...] sorted by weight descending
        """
        if self.method == 'lda':
            # LDA: model.components_ is K × vocab, row = topic distribution over words
            topic_dist = self.model.components_[topic_id]
        else:  # NMF
            topic_dist = self.model.components_[topic_id]
        
        top_indices = np.argsort(topic_dist)[::-1][:n_terms]
        top_terms = [(self.feature_names[i], topic_dist[i]) for i in top_indices]
        
        return top_terms
    
    def get_all_topic_descriptors(self, n_terms: int = 15) -> pd.DataFrame:
        """
        Get descriptors for all topics.
        
        Returns
        -------
        pd.DataFrame
            Columns: topic_id, top_terms (list), top_weights (list)
        """
        descriptors = []
        
        for topic_id in range(self.n_topics):
            terms_weights = self.get_top_terms(topic_id, n_terms)
            terms = [t for t, w in terms_weights]
            weights = [w for t, w in terms_weights]
            
            descriptors.append({
                'topic_id': topic_id + 1,  # 1-indexed for user-facing
                'top_terms': terms,
                'top_weights': weights,
                'top_terms_str': ', '.join(terms[:10])  # Human-readable
            })
        
        return pd.DataFrame(descriptors)
    
    def save(self, output_dir: Path):
        """Save model artifacts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn model
        with open(output_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open(output_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save theta as parquet
        theta_df = pd.DataFrame(
            self.theta,
            columns=[f'topic_{i+1}' for i in range(self.n_topics)]
        )
        theta_df.to_parquet(output_dir / 'theta.parquet', index=False)
        
        # Save metadata
        metadata = {
            'n_topics': self.n_topics,
            'method': self.method,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }
        
        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, input_dir: Path):
        """Load model artifacts."""
        input_dir = Path(input_dir)
        
        import json
        with open(input_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        with open(input_dir / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(input_dir / 'vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        theta_df = pd.read_parquet(input_dir / 'theta.parquet')
        theta = theta_df.values
        
        return cls(
            model=model,
            vectorizer=vectorizer,
            theta=theta,
            feature_names=metadata['feature_names'],
            n_topics=metadata['n_topics'],
            method=metadata['method'],
            random_state=metadata['random_state']
        )


def fit_topic_model(
    documents: List[str],
    n_topics: int = 20,
    method: Literal['nmf', 'lda'] = 'nmf',
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.7,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
    **model_kwargs
) -> TopicModelResults:
    """
    Fit a topic model (NMF or LDA) on documents.
    
    This is an STM-approximation for the 2-month validation.
    NMF is often preferred for interpretability and speed.
    
    Parameters
    ----------
    documents : list of str
        Input documents (cleaned text)
    n_topics : int
        Number of topics K
    method : {'nmf', 'lda'}
        Topic modeling method
        - 'nmf': Non-negative Matrix Factorization (fast, interpretable)
        - 'lda': Latent Dirichlet Allocation (probabilistic, slower)
    max_features : int
        Maximum vocabulary size
    min_df : int
        Minimum document frequency (ignore rare terms)
    max_df : float
        Maximum document frequency (ignore common terms)
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    verbose : int
        Verbosity level
    **model_kwargs
        Additional arguments for NMF/LDA model
        
    Returns
    -------
    TopicModelResults
        Fitted model with theta, descriptors, etc.
        
    Examples
    --------
    >>> docs = ["political debate about healthcare", "election campaign news", ...]
    >>> results = fit_topic_model(docs, n_topics=20, method='nmf')
    >>> descriptors = results.get_all_topic_descriptors()
    >>> theta = results.theta  # Document-topic proportions
    """
    print(f"Fitting {method.upper()} with {n_topics} topics on {len(documents)} documents...")
    
    # Vectorization
    if method == 'nmf':
        # NMF works better with TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams + bigrams
        )
    else:  # LDA
        # LDA requires raw counts
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"  Vocabulary size: {len(feature_names)}")
    print(f"  Document-term matrix: {X.shape}")
    
    # Fit topic model
    if method == 'nmf':
        # Extract max_iter from kwargs if present, otherwise use default
        max_iter = model_kwargs.pop('max_iter', 500)
        model = NMF(
            n_components=n_topics,
            random_state=random_state,
            init='nndsvda',  # Better initialization
            max_iter=max_iter,
            **model_kwargs
        )
    else:  # LDA
        # Extract max_iter from kwargs if present, otherwise use default
        max_iter = model_kwargs.pop('max_iter', 50)
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=max_iter,
            learning_method='batch',
            n_jobs=n_jobs,
            verbose=verbose,
            **model_kwargs
        )
    
    # Fit and get document-topic distributions
    theta = model.fit_transform(X)
    
    # Normalize theta (especially for NMF)
    if method == 'nmf':
        # Normalize rows to sum to 1
        row_sums = theta.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        theta = theta / row_sums
    
    print(f"  ✓ Model fitted")
    print(f"  Document-topic matrix (theta): {theta.shape}")
    
    return TopicModelResults(
        model=model,
        vectorizer=vectorizer,
        theta=theta,
        feature_names=feature_names.tolist(),
        n_topics=n_topics,
        method=method,
        random_state=random_state
    )


def get_topic_descriptors(
    results: TopicModelResults,
    n_terms: int = 15,
    include_weights: bool = True
) -> pd.DataFrame:
    """
    Extract topic descriptors (top terms).
    
    Convenience wrapper around TopicModelResults.get_all_topic_descriptors().
    
    Parameters
    ----------
    results : TopicModelResults
        Fitted model results
    n_terms : int
        Number of top terms per topic
    include_weights : bool
        Include term weights in output
        
    Returns
    -------
    pd.DataFrame
        Topic descriptors
    """
    return results.get_all_topic_descriptors(n_terms=n_terms)


def assign_topics_to_documents(
    results: TopicModelResults,
    documents_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    text_col: Optional[str] = None,
    theta_prefix: str = 'theta_topic_'
) -> pd.DataFrame:
    """
    Assign topic proportions to documents.
    
    Parameters
    ----------
    results : TopicModelResults
        Fitted model
    documents_df : pd.DataFrame
        Documents with doc_id (must align with theta row order)
    doc_id_col : str
        Document ID column name
    text_col : str, optional
        Text column (for validation, not used in assignment)
    theta_prefix : str
        Prefix for theta columns in output
        
    Returns
    -------
    pd.DataFrame
        Documents with added theta columns
        
    Notes
    -----
    This assumes documents_df rows align with results.theta rows
    (i.e., same order as input to fit_topic_model).
    
    For out-of-sample documents, use transform_new_documents instead.
    """
    if len(documents_df) != results.theta.shape[0]:
        raise ValueError(
            f"Document count mismatch: df has {len(documents_df)} rows, "
            f"theta has {results.theta.shape[0]} rows"
        )
    
    # Add theta columns
    theta_df = pd.DataFrame(
        results.theta,
        columns=[f'{theta_prefix}{i+1}' for i in range(results.n_topics)]
    )
    
    # Concatenate with original df
    output_df = pd.concat([documents_df.reset_index(drop=True), theta_df], axis=1)
    
    return output_df


def transform_new_documents(
    results: TopicModelResults,
    new_documents: List[str]
) -> np.ndarray:
    """
    Transform new documents to topic space (out-of-sample inference).
    
    Parameters
    ----------
    results : TopicModelResults
        Fitted model
    new_documents : list of str
        New documents to transform
        
    Returns
    -------
    np.ndarray
        Document-topic proportions (n_docs × K)
    """
    # Vectorize new documents
    X_new = results.vectorizer.transform(new_documents)
    
    # Transform to topic space
    theta_new = results.model.transform(X_new)
    
    # Normalize (for NMF)
    if results.method == 'nmf':
        row_sums = theta_new.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        theta_new = theta_new / row_sums
    
    return theta_new


def select_optimal_k(
    documents: List[str],
    k_range: range = range(10, 31, 5),
    n_runs: int = 3,
    method: Literal['nmf', 'lda'] = 'nmf',
    coherence_type: str = 'c_v',
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.7,
    random_state: int = 42,
    verbose: bool = True,
    **model_kwargs
) -> Dict:
    """
    Select optimal number of topics using coherence scores.
    
    Fits models with different K values, computes coherence for each,
    and returns results for analysis. Implements stability check via
    multiple runs per K value.
    
    Parameters
    ----------
    documents : list of str
        Input documents
    k_range : range or list of int
        K values to test
    n_runs : int
        Number of runs per K (for stability)
    method : {'nmf', 'lda'}
        Topic modeling method
    coherence_type : str
        Coherence measure ('c_v', 'u_mass', 'c_npmi')
    max_features : int
        Maximum vocabulary size
    min_df : int
        Minimum document frequency
    max_df : float
        Maximum document frequency
    random_state : int
        Base random seed (incremented per run)
    verbose : bool
        Print progress
    **model_kwargs
        Additional arguments for fit_topic_model
        
    Returns
    -------
    dict
        {
            'results': list of dicts with k, coherence_mean, coherence_std, coherence_scores
            'optimal_k': int (K with highest mean coherence)
            'all_models': list of TopicModelResults for each K (last run only)
        }
        
    Examples
    --------
    >>> selection = select_optimal_k(documents, k_range=range(10, 31, 5))
    >>> print(f"Optimal K: {selection['optimal_k']}")
    >>> results_df = pd.DataFrame(selection['results'])
    """
    from thesis_pipeline.topics.coherence import compute_topic_coherence
    from gensim.corpora import Dictionary
    
    # Tokenize documents once
    tokenized_docs = [doc.split() for doc in documents]
    
    results = []
    all_models = []
    
    if verbose:
        print(f"Exploring K range: {list(k_range)}")
        print(f"Runs per K: {n_runs}")
        print(f"Coherence: {coherence_type}")
        print()
    
    for k in k_range:
        if verbose:
            print(f"Testing K = {k}")
        
        k_scores = []
        
        for run in range(n_runs):
            # Fit model
            topic_model = fit_topic_model(
                documents=documents,
                n_topics=k,
                method=method,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                random_state=random_state + run,
                verbose=0,
                **model_kwargs
            )
            
            # Extract topics as list of term lists
            topics = []
            for topic_id in range(k):
                top_terms = topic_model.get_top_terms(topic_id, n_terms=10)
                topics.append([term for term, _ in top_terms])
            
            # Compute coherence using gensim
            dictionary = Dictionary(tokenized_docs)
            from gensim.models.coherencemodel import CoherenceModel
            cm = CoherenceModel(
                topics=topics,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence=coherence_type
            )
            coherence = cm.get_coherence()
            
            k_scores.append(coherence)
            
            if verbose:
                print(f"  Run {run+1}: coherence = {coherence:.4f}")
            
            # Save last model for this K
            if run == n_runs - 1:
                all_models.append(topic_model)
        
        # Store results
        results.append({
            'k': k,
            'coherence_mean': np.mean(k_scores),
            'coherence_std': np.std(k_scores),
            'coherence_scores': k_scores
        })
        
        if verbose:
            print(f"  Summary: {np.mean(k_scores):.4f} ± {np.std(k_scores):.4f}\n")
    
    # Find optimal K
    coherence_means = [r['coherence_mean'] for r in results]
    optimal_idx = np.argmax(coherence_means)
    optimal_k = results[optimal_idx]['k']
    
    if verbose:
        print("=" * 60)
        print("MODEL SELECTION SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"K = {r['k']:2d}: coherence = {r['coherence_mean']:.4f} ± {r['coherence_std']:.4f}")
        print(f"\nOptimal K (by coherence): {optimal_k}")
    
    return {
        'results': results,
        'optimal_k': optimal_k,
        'all_models': all_models
    }


# Example usage
if __name__ == '__main__':
    print("Topic Modeling Utilities Test")
    print("=" * 60)
    
    # Test data
    docs = [
        "healthcare reform debate congress bill",
        "election campaign presidential candidate votes",
        "supreme court ruling decision legal",
        "economic policy tax reform budget",
        "foreign policy military intervention troops",
        "climate change environmental regulation policy",
        "immigration reform border security enforcement",
        "healthcare insurance coverage affordable care",
        "election polls voting results turnout",
        "court decision constitutional rights amendment"
    ] * 10  # Repeat to have enough docs
    
    # Test 1: Fit NMF
    print("\nTest 1: Fit NMF model")
    results_nmf = fit_topic_model(
        docs,
        n_topics=5,
        method='nmf',
        max_features=100,
        min_df=1,
        random_state=42
    )
    
    print(f"\nTheta shape: {results_nmf.theta.shape}")
    print(f"Theta row sums (first 5): {results_nmf.theta[:5].sum(axis=1)}")
    
    # Test 2: Get descriptors
    print("\nTest 2: Topic descriptors")
    descriptors = results_nmf.get_all_topic_descriptors(n_terms=5)
    print(descriptors[['topic_id', 'top_terms_str']])
    
    # Test 3: Assign topics
    print("\nTest 3: Assign topics to documents")
    docs_df = pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(len(docs))],
        'text': docs
    })
    
    assigned = assign_topics_to_documents(results_nmf, docs_df)
    print(f"Assigned shape: {assigned.shape}")
    print(f"Theta columns: {[c for c in assigned.columns if 'theta' in c]}")
    
    # Test 4: Transform new documents
    print("\nTest 4: Transform new documents")
    new_docs = ["healthcare policy reform", "election voting campaign"]
    theta_new = transform_new_documents(results_nmf, new_docs)
    print(f"New theta shape: {theta_new.shape}")
    print(f"New theta row sums: {theta_new.sum(axis=1)}")
    
    print("\n✓ All tests passed!")
