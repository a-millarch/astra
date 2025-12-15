import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

from tsai.data.core import get_ts_dls
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy
from tsai.all import TSTabFusionTransformer, LabelSmoothingCrossEntropyFlat, Learner

from astra.utils import get_cfg, cfg, logger
from astra.evaluation.utils import save_figure, calculate_roc_auc_ci, calculate_average_precision_ci
from astra.visualize.evaluation import plot_evaluation

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve, 
    average_precision_score
)

#### Helper functions
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

def time_to_step(time_value, time_unit='min'):
    if time_unit == 'min':
        time_min = time_value
    elif time_unit == 'h':
        time_min = time_value * 60
    elif time_unit == 'D':
        time_min = time_value * 24 * 60
    else:
        raise ValueError("Unsupported time unit. Use 'min', 'h' or 'D'.")
    
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]
    
    for i, interval in enumerate(intervals):
        start_min = interval['start_h'] * 60
        end_min = interval['end_h'] * 60 if interval['end_h'] is not None else float('inf')
        if start_min < time_min <= end_min:
            offset_min = time_min - start_min
            step_offset = int(np.ceil(offset_min / interval['bin_min'])) - 1
            bins_cum = 0
            for j in range(i):
                duration_min = (intervals[j]['end_h'] - intervals[j]['start_h']) * 60
                bins_cum += duration_min // intervals[j]['bin_min']
            return bins_cum + step_offset
    return None

def step_to_time(step):
    intervals = [
        {'start_h': 0, 'end_h': 6, 'bin_min': 10},
        {'start_h': 6, 'end_h': 12, 'bin_min': 20},
        {'start_h': 12, 'end_h': 24, 'bin_min': 60},
        {'start_h': 24, 'end_h': 72, 'bin_min': 240},
        {'start_h': 72, 'end_h': 336, 'bin_min': 720},
        {'start_h': 336, 'end_h': 720, 'bin_min': 1440},
        {'start_h': 720, 'end_h': 2160, 'bin_min': 10080},
        {'start_h': 2160, 'end_h': None, 'bin_min': 43200},
    ]
    bins_cum = [0]
    for interval in intervals[:-1]:
        duration_min = (interval['end_h'] - interval['start_h']) * 60
        bins = duration_min // interval['bin_min']
        bins_cum.append(bins_cum[-1] + bins)
    
    for i in range(len(bins_cum) - 1):
        if bins_cum[i] <= step < bins_cum[i+1]:
            interval = intervals[i]
            step_offset = step - bins_cum[i]
            start_min = interval['start_h'] * 60
            return start_min + (step_offset + 1) * interval['bin_min']
    return None

def fill_zero(df, censor_t):
    cols_to_replace = [col for col in df.columns if col.isdigit() and int(col) > censor_t]
    df[cols_to_replace] = 0.0
    return df

def get_eval_mixed_dls(df, censor_t, data):
    cfg =get_cfg()
    
    ts_df = fill_zero(df.complete.copy(deep=True), censor_t)
    logger.debug(f'Preparing holdout dataloader for censor_t={censor_t}')
    tfms = data["tfms"]
    batch_tfms = data["batch_tfms"]
    procs = data["procs"]
    
    tX, ty = df2xy(ts_df, 
                   sample_col='PID', 
                   feat_col='FEATURE', 
                   data_cols=df.complete.columns[3:], 
                   target_col=df.target)
    ty = list(ty[:, 0].flatten())
    
    test_ts_dls = get_ts_dls(tX, ty, splits=None, tfms=tfms, batch_tfms=batch_tfms, 
                             bs=cfg["training"]["bs"], drop_last=False, shuffle=False)
    
    test_tab_dls = get_tabular_dls(df.tab_df, procs=procs, 
                                   cat_names=cfg["dataset"]["cat_cols"], 
                                   cont_names=cfg["dataset"]["num_cols"], 
                                   y_names=cfg["target"], 
                                   splits=None, 
                                   drop_last=False, shuffle=False, classes = data["classes"])
    return get_mixed_dls(test_ts_dls, test_tab_dls, bs=cfg["training"]["bs"], shuffle_valid=False)




def generate_time_thresholds(max_days=30, cut_hours=72, step_hours=1, step_days=1):
    thresholds = []
    for h in range(step_hours, cut_hours+1, step_hours):
        step = time_to_step(h, 'h')
        if step is not None:
            thresholds.append(step)
    start_day = int(np.ceil(cut_hours/24))
    for d in range(start_day+1, max_days+1, step_days):
        step = time_to_step(d, 'D')
        if step is not None:
            thresholds.append(step)
    return thresholds


def format_step_label(step):
    # Use your time_to_step function inverse
    time_min = step_to_time(step)  # returns minutes
    
    if time_min is None:
        return f"Step {step} (unknown)"
    
    if time_min < 60:
        return f"{int(time_min)} min"
    elif time_min < 24 * 60:
        hours = time_min / 60
        if hours.is_integer():
            hours = int(hours)
        return f"{hours} h"
    else:
        days = time_min / (24 * 60)
        if days.is_integer():
            days = int(days)
        return f"{days} day" + ("s" if days != 1 else "")
    

def evaluate_over_time(tsds, censor_ts, learn, data):
    results = []
    preds_over_time = []
    for censor_t in censor_ts:
        dls = get_eval_mixed_dls(tsds, censor_t, data)
        preds, targets = learn.get_preds(dl=dls.train)
        y_preds = preds[:, 1].numpy()
        ys = targets.numpy()

        if ys.sum() == 0 or ys.sum() == len(ys):
            continue

        auc, auc_lower, auc_upper = calculate_roc_auc_ci(ys, y_preds)
        ap, ap_lower, ap_upper = calculate_average_precision_ci(ys, y_preds)
        
        time_min = step_to_time(censor_t)
        if time_min is None:
            continue

        results.append({
            "time_min": time_min,
            "time_hours": time_min/60,
            "time_days": time_min/(24*60),
            "AUROC": auc,
            "AUROC_CI": (auc_lower, auc_upper),
            "AP": ap,
            "AP_CI": (ap_lower, ap_upper)
        })
        
        patient_ids = tsds.base.PID.values
        for pid, pred in zip(patient_ids, y_preds):
            preds_over_time.append({
                "PID": pid,
                "censor_t": censor_t,
                "time_min": time_min,
                "pred": float(pred)
            })

    return results, pd.DataFrame(preds_over_time)

### Model specific plot functions
def plot_multiple_evaluations(tsds, censor_ts, learn, data, labels=None):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    
    #colors = plt.cm.tab10(np.linspace(0, 1, len(censor_ts)))
    colors =['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B','#E377C2','#7F7F7F','#BCBD22','#17BECF']
    
    for i, censor_t in enumerate(censor_ts):
        dls = get_eval_mixed_dls(tsds, censor_t, data)
        preds, targets = learn.get_preds(dl=dls.train)
        
        y_preds = preds[:, 1]  # Assuming class 1 probability
        ys = targets
        
        # --- ROC ---
        fpr, tpr, _ = roc_curve(ys, y_preds)
        roc_auc = roc_auc_score(ys, y_preds)
        label = labels[i] if labels else f"censor_t={censor_t}"
        ax_roc.plot(fpr, tpr, color=colors[i % len(colors)], label=f"{label} (AUC={roc_auc:.3f})")
        
        # --- PR ---
       # PrecisionRecallDisplay.from_predictions(
       #     ys, y_preds, ax=ax_pr, name=label, color=colors[i % len(colors)]
       # )
        
        
        # --- PR ---
        precision, recall, _ = precision_recall_curve(ys, y_preds)
        auprc = average_precision_score(ys, y_preds)
        label = labels[i] if labels else f"censor_t={censor_t}"
        ax_pr.plot(recall, precision, color=colors[i % len(colors)], label=f"{label} (AUC={auprc:.3f})")
    
    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, c="grey")
    ax_roc.set_title("Holdout:\nReceiver Operator Characteristics (ROC)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid()
    ax_roc.legend(fontsize=9, title="Included time")  
    ax_roc.get_legend().get_title().set_fontsize(10)
    ax_roc.set_aspect('equal', adjustable='box')  # keep ROC plot square

    # PR formatting
    baseline = ys.sum() / len(ys)
    ax_pr.plot([0, 1], [baseline, baseline], 'k--', lw=1, c="grey")
    ax_pr.set_title("Holdout:\nPrecision Recall (PR)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid()
    ax_pr.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, title="Included time")  # outside right
    ax_pr.get_legend().get_title().set_fontsize(10)
    ax_pr.set_aspect('equal', adjustable='box') 

    # Adjust spacing to prevent overlap with PR legend
    fig.subplots_adjust(right=0.75)
    
    plt.savefig("models/figs/holdout_evaluation_multiple.png", dpi=1200)
    plt.show()
    return fig

def plot_time_metrics(results, cut_hours=72, max_days=30):
    times_h = np.array([r["time_hours"] for r in results])
    times_d = np.array([r["time_days"] for r in results])
    auroc_vals = np.array([r["AUROC"] for r in results])
    auroc_lower = np.array([r["AUROC_CI"][0] for r in results])
    auroc_upper = np.array([r["AUROC_CI"][1] for r in results])
    ap_vals = np.array([r["AP"] for r in results])
    ap_lower = np.array([r["AP_CI"][0] for r in results])
    ap_upper = np.array([r["AP_CI"][1] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    mask_cut = times_h <= cut_hours
    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals[mask_cut], auroc_lower[mask_cut], auroc_upper[mask_cut], 'o', "C0", "AUROC"),
        ("AP", ap_vals[mask_cut], ap_lower[mask_cut], ap_upper[mask_cut], 's', "C1", "AUPRC")
    ]:
        x = times_h[mask_cut]
        if len(x) > 0 and x[-1] < cut_hours:
            x_ext = np.append(x, cut_hours)
            vals_ext = np.append(vals, vals[-1])
            lower_ext = np.append(lower, lower[-1])
            upper_ext = np.append(upper, upper[-1])
        else:
            x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper
        ax1.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=3)
        ax1.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax1.set_xlabel("Time (hours)")
    ax1.set_xlim(0, cut_hours)
    ax1.set_xticks(np.arange(0, 73, 4))
    ax1.set_yticks(np.arange(0.0,1.1,0.1))
    ax1.set_ylabel("Score")
    ax1.set_title(f"A)")
    ax1.grid(True)
    ax1.legend()

    for metric, vals, lower, upper, marker, color, label in [
        ("AUROC", auroc_vals, auroc_lower, auroc_upper, 'o', "C0", "AUROC"),
        ("AP", ap_vals, ap_lower, ap_upper, 's', "C1", "AUPRC")
    ]:
        x = times_d
        if len(x) > 0 and x[-1] < max_days:
            x_ext = np.append(x, max_days)
            vals_ext = np.append(vals, vals[-1])
            lower_ext = np.append(lower, lower[-1])
            upper_ext = np.append(upper, upper[-1])
        else:
            x_ext, vals_ext, lower_ext, upper_ext = x, vals, lower, upper
        ax2.plot(x_ext, vals_ext, color=color, marker=marker, label=label, markersize=3)
        ax2.fill_between(x_ext, lower_ext, upper_ext, color=color, alpha=0.2)

    ax2.set_xlabel("Time (days)")
    ax2.set_xlim(0, max_days)
    ax2.set_xticks(np.arange(0, max_days+1, 5))
    ax2.set_yticks(np.arange(0.0,1.1,0.1))
    ax2.set_ylabel("Score")
    ax2.set_title("B)")
    ax2.grid(True)
    ax2.legend(loc='lower right')

    plt.tight_layout(pad=3.0)
    return fig


    
#### evaluation function
def run_eval(data, model_name: str, comprehensive_eval: bool = True):
    """Enhanced evaluation with time-dependent metrics"""
    mixed_dls = data["mixed_dls"]
    holdout = data["holdout"]
    num_cols = data["num_cols"]
    classes = data["classes"]
    
    # Load model
    backbone = TSTabFusionTransformer(
        c_in=mixed_dls.vars,
        c_out=2,
        seq_len=mixed_dls.len,
        classes=classes,
        cont_names=num_cols,
        n_layers=8,
        n_heads=8,
        d_model=64,
        fc_dropout=0.75,
        res_dropout=0.22,
        fc_mults=(0.3, 0.1),
    )
    loss_func = LabelSmoothingCrossEntropyFlat()
    learn = Learner(mixed_dls, backbone, loss_func=loss_func, metrics=None)
    learn = learn.load(model_name)
 # Original simple eval
    holdout_mixed_dls = data["holdout_mixed_dls"]
    preds, targs = learn.get_preds(dl=holdout_mixed_dls.train)
    evalplt = plot_evaluation(preds[:,1], targs,cfg["target"]) #change cfg["target"] to string for legend
    logger.info("ROC/PR plot done")
    
    if comprehensive_eval:
        logger.info("Running comprehensive time-dependent evaluation...")
        censor_thresholds = generate_time_thresholds(max_days=31, cut_hours=72)
        results, preds_df = evaluate_over_time(holdout, censor_thresholds, learn, data)
        
        # Save predictions
        os.makedirs('models/eval', exist_ok=True)
        preds_df.to_pickle(f'models/eval/preds_{model_name}.pkl')
        
        # Plot time metrics
        fig = plot_time_metrics(results, cut_hours=72)
        save_figure(fig, f"AUC_time_fig_{model_name}", save_dir='models/figs')
        
        logger.info(f"Comprehensive eval complete. Results: {len(results)} time points")
        
        # Plot multiple curves
        censor_thresholds = [
       # time_to_step(30, 'min'),
        time_to_step(60, 'min'),
        time_to_step(6, 'h'),
        time_to_step(12, 'h'),
         #time_to_step(24, 'h'),
        time_to_step(72, 'h'),
        time_to_step(7, 'D'),
        time_to_step(14, 'D'),
         #time_to_step(21, 'D'),   
        time_to_step(30, 'D')
        ]
        censor_thresholds.reverse()
        labels = [format_step_label(step) for step in censor_thresholds]
        multi_curve = plot_multiple_evaluations(holdout, censor_thresholds, learn, data,labels=labels)
       
        return results, preds_df