import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_and_compare_models(trained_models, X_test, y_test):
    results = []
    best_model_name = None
    best_f1 = 0

    print("\nEvaluating Models...\n")

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
        rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        results.append({
            "Model": name,
            "Accuracy": np.round(acc, 4),
            "Precision": np.round(prec, 4),
            "Recall": np.round(rec, 4),
            "F1 Score": np.round(f1, 4)
        })

        print(f"{name} Confusion Matrix")
        print(cm, "\n")

    metrics_df = pd.DataFrame(results)
    print("FINAL PERFORMANCE COMPARISON")
    print(metrics_df.to_string(index=False))
    print(f"\nBest Performing Model: {best_model_name} (F1 Score: {np.round(best_f1, 4)})\n")
    
    return best_model_name