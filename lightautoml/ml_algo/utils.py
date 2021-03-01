"""Tools for model training."""

from typing import Tuple, Optional, Callable

from log_calls import record_history

from .base import MLAlgo
from .tuning.base import ParamsTuner
from ..dataset.base import LAMLDataset
from ..utils.logging import get_logger
from ..validation.base import TrainValidIterator

import numpy as np
from scipy import optimize

logger = get_logger(__name__)


@record_history(enabled=False)
def tune_and_fit_predict(ml_algo: MLAlgo, params_tuner: ParamsTuner,
                         train_valid: TrainValidIterator,
                         force_calc: bool = True) -> Tuple[Optional[MLAlgo], Optional[LAMLDataset]]:
    """Tune new algorithm, fit on data and return algo and predictions.

    Args:
        ml_algo: ML algorithm that will be tuned.
        params_tuner: Tuner object.
        train_valid: Classic cv-iterator.
        force_calc: Flag if single fold of ml_algo should be calculated anyway.

    Returns:
        Tuple (BestMlAlgo, predictions).

    """

    timer = ml_algo.timer
    timer.start()
    single_fold_time = timer.estimate_folds_time(1)

    # if force_calc is False we check if it make sense to continue
    if not force_calc and ((single_fold_time is not None and single_fold_time > timer.time_left)
                           or timer.time_limit_exceeded()):
        return None, None

    if params_tuner.best_params is None:
        # this try/except clause was added because catboost died for some unexpected reason
        try:
            # TODO: Set some conditions to the tuner
            new_algo, preds = params_tuner.fit(ml_algo, train_valid)
        except Exception as e:
            logger.warning('Model {0} failed during params_tuner.fit call.\n\n{1}'.format(ml_algo.name, e))
            return None, None

        if preds is not None:
            return new_algo, preds

    if not force_calc and ((single_fold_time is not None and single_fold_time > timer.time_left)
                           or timer.time_limit_exceeded()):
        return None, None

    ml_algo.params = params_tuner.best_params
    # this try/except clause was added because catboost died for some unexpected reason
    try:
        preds = ml_algo.fit_predict(train_valid)
    except Exception as e:
        logger.warning('Model {0} failed during ml_algo.fit_predict call.\n\n{1}'.format(ml_algo.name, e))
        return None, None

    return ml_algo, preds

@record_history(enabled=False)
def find_best_constant(y_true: np.ndarray, weights: np.ndarray, loss: Callable):
    """Find best constant predictor.

    Args:
        y_true: Target values.
        weights: Weight of samples.
        loss: Loss function to optimize.

    """
    res = optimize.minimize_scalar(
        lambda p: loss(y_true, np.full_like(y_true, p), sample_weights=weights),
        bounds=(y_true.min(), y_true.max()),
        method='bounded'
    )
    p = res.x

    return p


def _pts_on_bound(loss: Callable, bmin, bmax, bins=20, add_last=False) -> Tuple[np.ndarray, np.ndarray]:
    """Find all values of loss function in bins.

    Args:
        loss: Scalar loss function.
        bmin: Left bound.
        bmax: Right bound.
        bins: Number of bins.
        add_last: Flag to add last point.

    Return:
        Tuple with points and its.

    """
    pts = np.linspace(bmin, bmax, bins + 1, dtype=np.float32)
    if not add_last:
        pts = pts[:-1]

    scores = np.array([loss(pt) for pt in pts])

    return pts, scores


@record_history(enabled=False)
def bounded_beam_search(y_true: np.ndarray, weights: np.ndarray,
                        loss: Callable, bins: int = 40, depth: int = 2,
                        beam_size: int = 1) -> float:
    """Beam search on segments.

    Args:
        y_true: Target values.
        weights: Weight of samples.
        loss: Scikit-like metric.
        bins: Number of bins to stratify.
        depth: Number from tree.
        beam_size: Number of stored values on every layer.

    Returns:
        Optimized value.

    """
    beam_size = min(beam_size, bins + 1)
    new_loss = lambda x: loss(
        y_true.astype(np.float32),
        np.full_like(y_true, x, dtype=np.float32),
        sample_weight=weights
    )

    bmax, bmin = y_true.max(), y_true.min()
    step_size = (bmax - bmin) / bins
    queue = [(bmin, bmax)]
    best_pts = []
    best_scores = []
    for d in range(depth):
        qsz = len(queue)
        for i, (bmin, bmax) in enumerate(queue):
            add_last = (i == qsz - 1)
            pts, scores = _pts_on_bound(new_loss, bmin, bmax, bins, add_last)
            ids = np.argsort(scores)[:beam_size]
            best_pts = np.concatenate((best_pts, pts[ids]))
            best_scores = np.concatenate((best_scores, scores[ids]))
        last_ids = np.argsort(best_scores)[:beam_size]
        best_pts = best_pts[last_ids]
        best_scores = best_scores[last_ids]
        if d != depth - 1:
            step_size /= 2
            queue = [(pt - step_size, pt + step_size) for pt in best_pts]
            best_pts = []
            best_scores = []

    idx = np.argmax(best_scores)
    best_pt = best_pts[idx]

    return best_pt


@record_history(enabled=False)
def find_baseline(y_true, weights, loss, task_name, fw_func):
    if task_name == 'multiclass':
        return None
    elif task_name == 'binary':
        val = bounded_beam_search(y_true, weights, loss)
        if 0 < val < 1:
            val = np.log(val / (1 - val))
            if fw_func is not None:
                val, _ = fw_func(val, 1)
            return val
        else:
            return None
    elif task_name == 'reg':
        val = find_best_constant(y_true, weights, loss)
        if fw_func is not None:
            val, _ = fw_func(val, 1)
        return val

