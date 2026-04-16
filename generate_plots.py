import os, joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack, csr_matrix

matplotlib.use('Agg')

artifacts = joblib.load('models/v2/training_artifacts.joblib')

test_df = artifacts['test_df']
numeric_cols = artifacts['numeric_feature_cols']
if 'numeric_scaler' in artifacts and artifacts['numeric_scaler'] is not None:
    X_num = artifacts['numeric_scaler'].transform(test_df[numeric_cols])
else:
    X_num = test_df[numeric_cols].values

if 'tfidf_vectorizer' in artifacts and artifacts['tfidf_vectorizer'] is not None:
    X_text = artifacts['tfidf_vectorizer'].transform(test_df['text_ml'])
    X_test = hstack([X_text, csr_matrix(X_num)])
else:
    X_test = X_num

y_test = test_df['label'].values

# Load models
models = {}
for name, p in artifacts['saved_model_paths'].items():
    models[name] = joblib.load(p)

os.makedirs('models/v2/plots', exist_ok=True)

MODEL_COLORS = {
    'logistic_regression': '#2563eb',
    'svm': '#16a34a',
    'random_forest': '#d97706',
    'neural_network': '#dc2626',
}

# 1. ROC Curve
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1)
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    color = MODEL_COLORS.get(name, 'black')
    label_name = name.replace("_", " ").title()
    ax.plot(fpr, tpr, lw=2, color=color, label=f'{label_name} (AUC={auc(fpr, tpr):.3f})')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (Test Dataset)')
ax.legend()
fig.savefig('models/v2/plots/roc_curves.png', bbox_inches='tight', dpi=150)
plt.close()

# 2. Confusion Matrices
fig, axes = plt.subplots(1, len(models), figsize=(4.5*len(models), 4.5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
fig.savefig('models/v2/plots/confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.close()
print('Plots generated successfully.')
