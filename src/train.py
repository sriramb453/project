from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import load_and_preprocess_data
from features import build_tfidf_features
from evaluate import evaluate_and_compare_models
from utils import save_artifact

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(3000)
   
    vectorizer, X_train_dense, X_test_dense = build_tfidf_features(X_train, X_test)
    
    models = {
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, thread_count=1)
    }

    for name, model in models.items():
        model.fit(X_train_dense, y_train)
     
    best_model_name = evaluate_and_compare_models(models, X_test_dense, y_test)
    
    save_artifact(models[best_model_name], "best_model.joblib", directory="..")
    save_artifact(vectorizer, "tfidf_vectorizer.joblib", directory="..")
    
    print("Pipeline complete!")

if __name__ == "__main__":
    main()