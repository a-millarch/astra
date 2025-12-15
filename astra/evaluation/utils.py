import os
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from scipy import stats

from sklearn.metrics import roc_auc_score, average_precision_score
from astra.utils import logger


def save_figure(fig, filename, save_dir='reports/studyfigs'):
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=1200, bbox_inches='tight')
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=1200, bbox_inches='tight')
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    base64_path = os.path.join(save_dir, f'{filename}_base64.txt')
    with open(base64_path, 'w') as f:
        f.write(base64_image)
    
    plt.close(fig)
    logger.info(f"Saved: {png_path}, {base64_path}")
    return png_path, base64_path



def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(predictions)
    ground_truth = ground_truth[order]
    predictions = predictions[order]
    n_pos = np.sum(ground_truth)
    n_neg = len(ground_truth) - n_pos
    pos_ranks = np.where(ground_truth == 1)[0] + 1
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    v01 = (auc / (2 - auc) - auc ** 2) / n_neg
    v10 = (2 * auc ** 2 / (1 + auc) - auc ** 2) / n_pos
    return v01 + v10

def calculate_roc_auc_ci(y_true, y_pred, alpha=0.95):
    auc = roc_auc_score(y_true, y_pred)
    auc_var = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_var)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    ci[ci < 0] = 0
    return auc, ci[0], ci[1]

def calculate_average_precision_ci(y_true, y_pred, alpha=0.95, n_bootstraps=1000):
    ap = average_precision_score(y_true, y_pred)
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = average_precision_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.sort(np.array(bootstrapped_scores))
    ci_lower = sorted_scores[int((1.0-alpha)/2 * len(sorted_scores))]
    ci_upper = sorted_scores[int((1.0+alpha)/2 * len(sorted_scores))]
    return ap, float(ci_lower), float(ci_upper)

