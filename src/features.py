from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_features(X_train, X_test, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    X_train_dense = vectorizer.fit_transform(X_train).toarray()
    X_test_dense = vectorizer.transform(X_test).toarray()
    
    return vectorizer, X_train_dense, X_test_dense