from tsai.all import TensorMultiCategory, Categorize, BCEWithLogitsLossFlat
from fastai.callback.all import *
from fastcore.foundation import L
from matplotlib import pyplot as plt
from fastai.callback.core import Callback
from fastcore.basics import store_attr, range_of
import numpy as np

from fastai.callback.core import TrainEvalCallback, CancelValidException


class SkipValidationCallback(TrainEvalCallback):
    def before_validate(self):
        raise CancelValidException()

def pv(text, verbose):
    if verbose:
        print(text)


class CSaveModel(TrackerCallback):
    def __init__(
        self,
        monitor="valid_loss",
        comp=None,
        min_delta=0.0,
        fname="model",
        every_epoch=False,
        at_end=False,
        with_opt=False,
        reset_on_fit=True,
        verbose=False,
    ):
        super().__init__(
            monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit
        )
        assert not (
            every_epoch and at_end
        ), "every_epoch and at_end cannot both be set to True"
        self.last_saved_path = None
        self.best_epoch = None
        store_attr("fname,every_epoch,at_end,with_opt,verbose")

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        # Compare the value monitored to its best score and save if best.
        if self.every_epoch:
            if (self.epoch % self.every_epoch) == 0:
                self._save(f"{self.fname}_{self.epoch}")
        else:  # every improvement
            super().after_epoch()
            if self.new_best:
                self.best_epoch = self.epoch  # Store the best epoch
                pv(
                    f"Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.",
                    self.verbose,
                )
                self._save(f"{self.fname}")

    def after_fit(self, **kwargs):
        # Load the best model and print the best epoch.
        if self.at_end:
            self._save(f"{self.fname}")
        elif not self.every_epoch:
            self.learn.load(f"{self.fname}", with_opt=self.with_opt)
            if self.best_epoch is not None:
                print(f"Best model was from epoch {self.best_epoch}")


def save_loss_plot(learn, save_path: str):
    skip_start = 5
    plt.plot(
        list(range(skip_start, len(learn.recorder.losses))),
        learn.recorder.losses[skip_start:],
        label="train",
    )
    idx = (np.array(learn.recorder.iters) < skip_start).sum()
    valid_col = learn.recorder.metric_names.index("valid_loss") - 1
    plt.plot(learn.recorder.iters[idx:], L(learn.recorder.values[idx:]).itemgot(valid_col), label="valid")  # type: ignore
    plt.legend()
    plt.savefig(save_path)
    plt.close()


class CustomTSMultiLabelClassification(Categorize):
    "Reversible combined transform of multi-category strings to one-hot encoded `vocab` id"

    loss_func, order = BCEWithLogitsLossFlat(), 1

    def __init__(self, c=None, vocab=None, add_na=False, sort=True):
        super().__init__(vocab=vocab, add_na=add_na, sort=sort)
        self.c = c

    def setups(self, dsets):
        if not dsets:
            return

    def encodes(self, o):
        return TensorMultiCategory(o)




class TrainingShowGraph(Callback):
    "(Modified) Update a graph of training and validation loss"

    order, run_valid = 65, False
    names = ["train", "valid"]

    def __init__(
        self, plot_metrics: bool = True, final_losses: bool = True, perc: float = 0.5
    ):
        store_attr()

    def before_fit(self):
        self.run = not hasattr(self.learn, "lr_finder") and not hasattr(
            self, "gather_preds"
        )
        if not (self.run):
            return
        self.nb_batches = []
        self.learn.recorder.loss_idxs = [i for i, n in enumerate(self.learn.recorder.metric_names[1:-1]) if "loss" in n]  # type: ignore
        _metrics_info = [(i, n) for i, n in enumerate(self.learn.recorder.metric_names[1:-1]) if "loss" not in n]  # type: ignore

        if len(_metrics_info) > 0:
            self.metrics_idxs, self.metrics_names = list(zip(*_metrics_info))
        else:
            self.metrics_idxs, self.metrics_names = None, None

    def after_train(self):
        self.nb_batches.append(self.train_iter - 1)

    def after_epoch(self):
        "Plot validation loss in the pbar graph"
        if not self.nb_batches:
            return
        rec = self.learn.recorder  # type: ignore
        if self.epoch == 0:
            self.rec_start = len(rec.losses)
        iters = range_of(rec.losses)
        all_losses = rec.losses if self.epoch == 0 else rec.losses[self.rec_start - 1 :]

        modified_recorder_values = [sublist[:-1] for sublist in self.learn.recorder.values]  # type: ignore

        val_losses = np.stack(modified_recorder_values)[:, self.learn.recorder.loss_idxs[-1]].tolist()  # type: ignore
        if rec.valid_metrics and val_losses[0] is not None:
            all_losses = all_losses + val_losses
        else:
            val_losses = [None] * len(iters)
        y_min, y_max = min(all_losses), max(all_losses)
        margin = (y_max - y_min) * 0.05
        x_bounds = (0, len(rec.losses) - 1)
        y_bounds = (y_min - margin, y_max + margin)
        self.update_graph(
            [(iters, rec.losses), (self.nb_batches, val_losses)], x_bounds, y_bounds
        )

    def after_fit(self):
        if hasattr(self, "graph_ax"):
            plt.close(self.graph_ax.figure)
        if self.plot_metrics:
            self.learn.plot_metrics(final_losses=self.final_losses, perc=self.perc)  # type: ignore

    def update_graph(self, graphs, x_bounds=None, y_bounds=None, figsize=(6, 4)):
        if not hasattr(self, "graph_fig"):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)  # type: ignore
        self.graph_ax.clear()
        if len(self.names) < len(graphs):
            self.names += [""] * (len(graphs) - len(self.names))
        for g, n in zip(graphs, self.names):
            if g[1] == [None] * len(g[1]):
                continue
            self.graph_ax.plot(*g, label=n)
        self.graph_ax.legend(loc="upper right")
        self.graph_ax.grid(color="gainsboro", linewidth=0.5)
        if x_bounds is not None:
            self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None:
            self.graph_ax.set_ylim(*y_bounds)
        self.graph_ax.set_title(f"Losses\nepoch: {self.epoch +1}/{self.n_epoch}")
        self.graph_out.update(self.graph_ax.figure)