# -------------------------------------------------------------------
# Import required packages and setup environment
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
import math
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
from lifelines import KaplanMeierFitter, statistics
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError
from shap import Explanation
from shap.plots import colors
from shap.plots._labels import labels
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    merge_nodes,
    sort_inds,
)
from typing import Literal
import scipy
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import torch
from matplotlib import colormaps
import scipy.stats as st
from matplotlib.figure import Figure
from typing import Any, Callable

import shap
from loess import loess_1d
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.util import Surv
from numpy.linalg import LinAlgError
from org import excel, write_to_file, pdf, save_as_svg # Local functions for organizing the output
from datetime import datetime

# -------------------------------------------------------------------
# Global parameters and constants
# -------------------------------------------------------------------
date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') # Date and time to save output
REPETITIONS = 20
MICE_DATASETS = 10
FOLDS = 5
SEED = 42
nbr_of_trials = 1
RESAMPLE = 10
n_bootstraps = 100
size_bootstrap = 100
NAME = 'survival_analysis_shap_hr_figs'
rng = np.random.default_rng(seed=SEED)

# Mapping of internal variable names to human-readable names for plots
change_names = {'clinical_frailty_scale': 'CFS', 'leukocytes': 'Leukocytes', 'albumin': 'Albumin', 'ngal': 'NGAL',
                'ddimer': 'D-dimer', 'il6': 'IL-6', 'age_at_admission': 'Age', 'meds_per_pat': '#Meds/Patient',
                'time_at_hospital_before_admission': '#Days in Hosp. before ICU', 'no_CPR': 'DNR order',
                'thrombocytes': 'WBC', 'neutrofila': 'Neutrophils', 'A02BC': 'PPI medication',
                'cystatin_c': 'Cystatin C',
                'smoker': 'Smoker',
                'congestive_heart_failure': 'Congestive heart failure',
                'cancer': 'Cancer',
                'hypertension': 'Hypertension',
                'myocardial_infarction': 'Myocardial infarction',
                'peripheral_vascular_disease': 'Peripheral vascular disease',
                'cerebrovascular_insult': 'Cerebrovascular insult',
                'chronic_pulmonary_disease': 'Chronic pulmonary disease',
                'rheumatic_disease': 'Rheumatic disease',
                'peptic_ulcer_disease': 'Peptic ulcer disease',
                'diabetes_mellitus_uncompl': 'Diabetes mellitus (uncompl.)',
                'diabetes_mellitus_compl': 'Diabetes mellitus (compl.)',
                'chronic_kidney_disease': 'Chronic kidney disease',
                'imv_within_24h': 'IMV within 24h from ICU admission',
                'noradrenaline': 'Noradrenaline',
                'sex_male': 'Sex',
                'GCS': 'Glasgow Coma Scale',
                'cardiovascular_map': 'Cardiovascular mean arterial pressure',
                'creatinine': 'Creatinine',
                'bilirubin': 'Bilirubin',
                'ferritin': 'Ferritin',
                'laktat': 'Lactate',
                'pct': 'Procalcitonin',
                'pao2': 'Pa0$_2$',
                'paco2': 'PaC0$_2$',
                'aB_pH': 'pH (arterial blood)',
                'lymphocytes': 'Lymphocytes',
                'body_temperature': 'Body temperature (\N{DEGREE SIGN}C)',
                'heart_frequency': 'Heart frequency (Hz)',
                'systolic_blood_pressure': 'Systolic blood pressure (mmHg)',
                'arterial_oxygen_tension': 'Arterial oxygen tension (mmHg)',
                'symptomatic_days': '#Symptomatic days',
                'habitual_creatinine': 'Habitual creatinine',
                'ICU_strain': 'ICU burden',
                'endostatin': 'Endostatin',
                'vcam': 'VCAM',
                'icam': 'ICAM',
                'calprotectin': 'Calprotectin',
                'pf_quotient': 'PaC0$_2$/FiO$_2$ Ratio',
                'survival_days_from_icu_start': '#Survival days from ICU admission',
                'mort': 'Dead within one year',
                'study_month': 'Study month'
                }  # 'NOAK', 'CRP', 'BMI', 'CCI', 'ACEi', 'ARB'

# -------------------------------------------------------------------
# Utility Function to rename variables
# -------------------------------------------------------------------
def plot_names(variables_to_rename, feat_name, feat_names):
    for old, new in variables_to_rename.items():
        if feat_name == old:
            feat_name = new
        elif old in feat_names:
            feat_names = [new if x == old else x for x in feat_names]
    return feat_name, feat_names

# -------------------------------------------------------------------
# SHAP label dictionary for consistent plotting and naming
# -------------------------------------------------------------------
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}


# -------------------------------------------------------------------
# Customization of the SHAP beeswarm plot to use on SHAP values collected in a training and internal validation loop
# -------------------------------------------------------------------
def beeswarm(
        shapvalues,  #  (n, 13)
        featurevalues: np.ndarray | None = None,  # (n, 13)
        featurenames: list | None = None,  #  (list of length 13)
        shap_values_shape: tuple | None = None,  # (n, 13)
        max_display: int | None = 14,
        order=Explanation.abs.mean(0),
        cluster_threshold=0.5,
        partition_tree=None,
        color=None,
        axis_color="#333333",
        alpha: float = 1.0,
        ax: plt.Axes | None = None,
        show: bool = True,
        log_scale: bool = False,
        color_bar: bool = True,
        s: float = 16,
        plot_size: Literal["auto"] | float | tuple[float, float] | None = "auto",
        color_bar_label: str = labels["FEATURE_VALUE"],
        group_remaining_features: bool = True,
):
    """
    Create a SHAP beeswarm plot for visualizing feature importance and the distribution
    of SHAP values across all samples.

    This function replicates SHAP's standard beeswarm visualization with added flexibility.
    It takes the SHAP values directly instead of an Explanation object to be used for only internal validation data (unseen during training).
    It shows the magnitude and direction of SHAP values for each feature and uses color
    to represent the original feature value.

    Parameters
    ----------
    shapvalues : np.ndarray
        The matrix of SHAP values with shape (n_samples, n_features). For SHAP
        `Explanation` objects, this is typically `explanation.values[:, :, 1]`.
    featurevalues : np.ndarray or None, optional
        The original input feature values corresponding to the SHAP values.
        If None, feature coloring will be disabled. Default is None.
    featurenames : list or None, optional
        Names of the features. If None, generic names (e.g. "Feature 0") are used.
    shap_values_shape : tuple or None, optional
        Shape of the SHAP values array (used for validation).
    max_display : int or None, optional
        Maximum number of top features to display in the plot. Default is 14.
    order : callable or np.ndarray, optional
        Function or array defining feature display order. Default is
        `Explanation.abs.mean(0)` (sorted by mean absolute SHAP value).
    cluster_threshold : float, optional
        Distance threshold for merging clustered features. Default is 0.5.
    partition_tree : optional
        Precomputed clustering tree (output of `scipy.cluster.hierarchy.linkage`).
        Default is None.
    color : str or matplotlib Colormap or None, optional
        Color map or single color for scatter points. Default depends on data presence.
    axis_color : str, optional
        Color of the plot axes. Default is "#333333".
    alpha : float, optional
        Opacity of the scatter points. Default is 1.0.
    ax : matplotlib.axes.Axes or None, optional
        Matplotlib axis to draw the plot on. If None, creates a new axis.
    show : bool, optional
        Whether to immediately display the plot. Default is True.
    log_scale : bool, optional
        Whether to use a symlog scale for the x-axis. Default is False.
    color_bar : bool, optional
        Whether to include a color bar representing feature values. Default is True.
    s : float, optional
        Size of the scatter points. Default is 16.
    plot_size : {'auto', float, tuple of float, None}, optional
        Plot size in inches or 'auto' to scale by number of features. Default is 'auto'.
    color_bar_label : str, optional
        Label for the color bar. Default is `labels["FEATURE_VALUE"]`.
    group_remaining_features : bool, optional
        Whether to group features beyond `max_display` into a single “sum of others”
        feature. Default is True.

    Raises
    ------
    ValueError
        If `shapvalues` has invalid shape or if incompatible `ax` and `plot_size`
        arguments are passed.
    DimensionError
        If feature and SHAP value matrices have inconsistent shapes.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axis containing the plot, or None if `show=True`.

    Notes
    -----
    - The beeswarm plot combines a summary of feature importance with the
      distribution of SHAP values for each feature.
    - Each dot corresponds to a single prediction's SHAP value for a given feature.
    - Color encodes the feature’s raw value (e.g., red = high, blue = low).
    - Features are ordered by average impact on the model output magnitude.

    """
    # Change the variable names for the plot
    feat_name = 'random_str_since_it_is_not_applicable'
    feat_names = featurenames
    _, feat_names = plot_names(change_names, feat_name, feat_names)
    print(feat_names)

    sv_shape = shap_values_shape
    if len(sv_shape) == 1:
        emsg = (
            "The beeswarm plot does not support plotting a single instance, please pass "
            "an explanation matrix with many instances!"
        )
        raise ValueError(emsg)
    elif len(sv_shape) > 2:
        emsg = (
            "The beeswarm plot does not support plotting explanations with instances that have more than one dimension!"
        )
        raise ValueError(emsg)

    if ax and plot_size:
        emsg = (
            "The beeswarm plot does not support passing an axis and adjusting the plot size. "
            "To adjust the size of the plot, set plot_size to None and adjust the size on the original figure the axes was part of"
        )
        raise ValueError(emsg)

    values = shapvalues
    features = featurevalues
    if scipy.sparse.issparse(features):
        features = features.toarray()
    feature_names = feat_names  # rephrased names

    order = convert_ordering(order, values)

    # default color:
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    assert isinstance(fig, Figure)  # type narrowing for mypy

    if log_scale:
        ax.set_xscale("symlog")

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()  # SHAP values
    while True:
        feature_order = convert_ordering(order,
                                         Explanation(np.abs(values)))
        if partition_tree is not None:
            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the partition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if (
                    max_display < len(feature_order)
                    and dist[feature_order[max_display - 1], feature_order[max_display - 2]] <= cluster_threshold
            ):
                # values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for _ in range(len(values)):
                    values[:, ind1] += values[:, ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for inds in orig_inds:
        if len(inds) == 1:
            feature_names_new.append(feat_names[inds[0]])  # feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feat_names[i] for i in
                                                 inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(
                f"{feat_names[inds[max_ind]]} + {len(inds) - 1} other features")
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    include_grouped_remaining = num_features < len(values[0]) and group_remaining_features
    if include_grouped_remaining:
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features - 1, len(values[0]))])
        values[:, feature_order[num_features - 1]] = np.sum(
            [values[:, feature_order[i]] for i in range(num_features - 1, len(values[0]))], 0
        )

    # build our y-tick labels
    yticklabels = [str(feature_names[i]) + ' (' + str(np.round(np.mean(np.abs(values[:, i])), 2)) + ')' for i in feature_inds]  # Adding the mean absolute SHAP value for each feature
    if include_grouped_remaining:
        yticklabels[-1] = f"Sum of {num_cut} other features"

    row_height = 0.4
    if plot_size == "auto":
        fig.set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
    elif isinstance(plot_size, (list, tuple)):
        fig.set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        fig.set_size_inches(8, min(len(feature_order), max_display) * plot_size + 1.5)
    ax.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]  # Shap values
        fvalues = None if features is None else features[:, i]  # original values
        f_inds = np.arange(len(shaps))
        np.random.shuffle(f_inds)
        if fvalues is not None:
            fvalues = fvalues[f_inds]
        shaps = shaps[f_inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:  # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
        except Exception:
            colored_feature = False
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds_ = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds_:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if safe_isinstance(color, "matplotlib.colors.Colormap") and fvalues is not None and colored_feature is True:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax

            if features is not None and features.shape[0] != len(shaps):
                emsg = "Feature and SHAP matrices must have the same number of rows!"
                raise DimensionError(emsg)

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            ax.scatter(
                shaps[nan_mask],
                pos + ys[nan_mask],
                color="#777777",
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(
                shaps[np.invert(nan_mask)],
                pos + ys[np.invert(nan_mask)],
                cmap=color,
                vmin=vmin,
                vmax=vmax,
                s=s,
                c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )
        else:
            if safe_isinstance(color, "matplotlib.colors.Colormap") and hasattr(color, "colors"):
                color = color.colors
            ax.scatter(
                shaps,
                pos + ys,
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                color=color if colored_feature else "#777777",
                rasterized=len(shaps) > 500,
            )

    # draw the color bar
    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_inds)), list(reversed(yticklabels)), fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-1, len(feature_inds))
    ax.set_xlabel(labels["VALUE"], fontsize=13)
    if show:
        plt.show()
    else:
        return ax

# -------------------------------------------------------------------
# Scatter plots with a LOESS curve for continuous variables
# -------------------------------------------------------------------
def plot_shap_feature(
        shap_values: Any,
        feature_values: pd.DataFrame,
        feature_name: int | str,
        feature_names: list,
        hr_ci: pd.DataFrame,
        n_plots: int = 1
) -> pd.DataFrame:
    """
    Generates SHAP dependency plots for a given feature and estimates hazard ratios (HR)
    with confidence intervals using LOESS smoothing and bootstrapping.

    This function visualizes the relationship between feature values and SHAP values
    for one-year mortality risk, optionally colored by additional features. It handles
    binary features, continuous features, and log-transformed features with special
    treatment for variables like '#Days in Hosp. before ICU' and 'IL-6'.
    Confidence intervals are computed using bootstrapped LOESS fits. Reference points
    for hazard ratios are annotated for certain variables.

    Parameters
    ----------
    shap_values : Any
        Array-like or DataFrame of SHAP values for each feature.
    feature_values : pd.DataFrame
        DataFrame containing the feature values corresponding to the SHAP values.
    feature_name : int | str
        Name or index of the feature to plot.
    feature_names : list
        List of feature names used for coloring or labeling multiple dependency plots.
    hr_ci : pd.DataFrame
        DataFrame to store reference values and confidence intervals.
    n_plots : int, default=1
        Number of features to color by in the scatter plots.

    Returns
    -------
    pd.DataFrame
        Updated `hr_ci` DataFrame containing LOESS-based SHAP estimates and confidence
        intervals at clinically appropriate values for hazard ratios.

    Notes
    -----
    - Binary features are plotted without LOESS smoothing.
    - Continuous features are smoothed using LOESS with bootstrapped confidence intervals.
    - Special transformations are applied for features requiring log-scaling.
    - Red circles mark key reference values in the plots for visual emphasis.

    """

    # Change the variable names for the plot
    feat_name = feature_name
    feat_names = feature_names
    feat_name, feat_names = plot_names(change_names, feat_name, feat_names)
    print(feat_name)
    print(feat_names)

    # Color dots given another feature's values
    for i in range(n_plots):
        coloring_feature = feature_names[i]
        col_data = feature_values[coloring_feature]
        print('coloring feature', coloring_feature)

        # No added LOESS curve for a binary variable
        if feature_values[feature_name].isin([0, 1]).all():
            plt.figure()
            plt.axhline(y=0, color='k', linestyle='dashed')
            sc = plt.scatter(feature_values[feature_name], shap_values[feature_name],
                             cmap=colormaps['coolwarm'], c=col_data, s=40, label=f"SHAP value for {feat_name}")
            # Labels and colorbar
            plt.xlabel(feat_name)
            plt.ylabel("SHAP value")
            plt.title(
                "Impact on one-year mortality risk",
            )
            cbar = plt.colorbar(sc)
            cbar.set_label(feat_names[i])
            plt.grid(True)
            plt.tight_layout()
            save_as_svg(date_time, 'survival_dep_' + str(feature_name) +
                        '_' + str(coloring_feature), subfolder='dependency_plots')

            continue  # plot a dependency plot contra the next feature

        # A version of logarithmizing the feature values for two variables
        elif feat_name == '#Days in Hosp. before ICU' or feat_name == 'IL-6':
            if feat_name == '#Days in Hosp. before ICU':
                x_label_name = 'log(' + str(feat_name) + ' + 1)'
                log_feature = np.log(feature_values[feature_name] + 1)  # +1 to avoid zero
            elif feat_name == 'IL-6':
                x_label_name = 'log(' + str(feat_name) + ')'
                log_feature = np.log(feature_values[feature_name])
            raw_feature = np.asarray(log_feature).ravel()
            raw_shap = np.asarray(shap_values[feature_name]).ravel()

            # Grid for smoothing
            x_grid = np.linspace(np.min(raw_feature), np.max(raw_feature), len(raw_feature))
            x_grid = np.asarray(x_grid).ravel()

            # Apply loess
            xout, yout, wout = loess_1d.loess_1d(raw_feature, raw_shap,
                                                 x_grid, degree=2, frac=0.6)

            # Bootstrap LOESS to estimate confidence interval
            boot_loess = []

            all_indices = feature_values.index.to_list()
            for ii in range(n_bootstraps):
                if int(len(feature_values.index) / size_bootstrap) < len(all_indices):
                    indices = rng.choice(all_indices,
                                         size=int(len(all_indices)),
                                         replace=False)
                else:
                    indices = rng.choice(all_indices,
                                         size=int(len(feature_values.index) / 10),
                                         replace=False)  # to not get two of the same value that leads to error in division
                    all_indices = [e for e in all_indices if e not in indices.tolist()]

                x_boot = log_feature[indices]
                raw_x = np.asarray(x_boot).ravel()
                y_boot = shap_values[feature_name][indices]
                raw_y = np.asarray(y_boot).ravel()
                try:
                    xout_boot, yout_boot, wout_boot = loess_1d.loess_1d(raw_x, raw_y, x_grid, degree=2,
                                                                    frac=0.6)  # Larger fraction --> smoother curve
                except LinAlgError as linalg:
                    print("Linear algebra error:", linalg)
                    continue


                boot_loess.append(yout_boot)

            boot_loess = np.array(boot_loess)
            conf_interval = st.t.interval(0.95, len(boot_loess) - 1, loc=yout, scale=st.sem(boot_loess))

            # For IL-6, add marking of the reference values
            fig, ax = plt.subplots()
            if feat_name == 'IL-6':
                plt.xlim(right=8.5)
                plt.ylim(-4, 4)
                # Draw a round circle in display space, even with unequal axes
                circle_size_pts = 400  # Size in points

                da = DrawingArea(2 * circle_size_pts, 2 * circle_size_pts, 0, 0)

                # xx1 circle
                x_grid_rounded = [round(xs) for xs in x_grid]
                xx1idx = [i for i, x in enumerate(x_grid_rounded) if x == 4]  # idx for xx1
                if len(xx1idx) > 0:
                    circlexx1 = plt.Circle((circle_size_pts, circle_size_pts),
                                         radius=circle_size_pts / 30,
                                         facecolor='none', edgecolor='red', linewidth=2)
                    da.add_artist(circlexx1)
                    abxx1 = AnnotationBbox(da, (4, yout[xx1idx[10]]),
                                         xycoords='data',
                                         frameon=False,
                                         box_alignment=(0.5, 0.5))  # Center the circle

                    ax.add_artist(abxx1)

                # xx2 circle
                xx2idx = [i for i, x in enumerate(x_grid_rounded) if x == 7]  # idx for xx2
                if len(xx2idx) > 0:
                    circlexx2 = plt.Circle((circle_size_pts, circle_size_pts),
                                           radius=circle_size_pts / 30,
                                           facecolor='none', edgecolor='red', linewidth=2)
                    da.add_artist(circlexx2)
                    abxx2 = AnnotationBbox(da, (7, yout[xx2idx[7]]),
                                           xycoords='data',
                                           frameon=False,
                                           box_alignment=(0.5, 0.5))  # Center the circle

                    ax.add_artist(abxx2)

            # Scatter plots with LOESS
            plt.axhline(y=0, color='k', linestyle='dashed')
            sc = plt.scatter(log_feature, shap_values[feature_name],
                             cmap=colormaps['coolwarm'], c=col_data, s=40, label=f"SHAP value for {feat_name}")
            plt.plot(x_grid, yout, color='k', label=f"LOESS for SHAP for {feat_name}")
            plt.fill_between(x_grid, conf_interval[0], conf_interval[1], color='k', alpha=0.08)
            # Labels and colorbar
            plt.xlabel(x_label_name)
            plt.ylabel("SHAP value")
            plt.title(
                "Variable effect on SHAP value for one-year mortality risk"
            )
            if feat_name == '#Days in Hosp. before ICU':
                plt.xlim(right=3.5)
                plt.ylim(-4, 4)
            cbar = plt.colorbar(sc)
            cbar.set_label(feat_names[i])
            plt.grid(True)
            plt.tight_layout()
            save_as_svg(date_time, 'survival_dep_log(' + str(feature_name) +
                        ')_' + str(coloring_feature), subfolder='dependency_plots')

        ### For all continuous variables
        raw_feature = np.asarray(feature_values[feature_name]).ravel() # Convert to NumPy arrays
        raw_shap = np.asarray(shap_values[feature_name]).ravel() # Convert to NumPy arrays

        # Grid for smoothing
        x_grid = np.linspace(np.min(raw_feature), np.max(raw_feature), len(raw_feature))
        x_grid = np.asarray(x_grid).ravel()

        # Apply loess
        xout, yout, wout = loess_1d.loess_1d(raw_feature, raw_shap,
                                             x_grid, degree=2, frac=0.6)

        # Bootstrap LOESS to estimate confidence interval
        boot_loess = []
        all_indices = feature_values.index.to_list()
        for ii in range(n_bootstraps):
            if int(len(feature_values.index) / size_bootstrap) < len(all_indices):
                indices = rng.choice(all_indices,
                                     size=int(len(all_indices)),
                                     replace=False) # No replacement to keep the values uncorrelated
            else:
                indices = rng.choice(all_indices, #feature_values.index,
                                     size=int(len(feature_values.index) / 10),
                                     replace=False) # No replacement to keep the values uncorrelated
                all_indices = [e for e in all_indices if e not in indices.tolist()]

            x_boot = feature_values[feature_name][indices]
            raw_x = np.asarray(x_boot).ravel()
            y_boot = shap_values[feature_name][indices]
            raw_y = np.asarray(y_boot).ravel()
            xout_boot, yout_boot, wout_boot = loess_1d.loess_1d(raw_x, raw_y, x_grid, degree=2,
                                                                frac=0.6)  # Larger fraction --> smoother curve
            boot_loess.append(yout_boot)

        boot_loess = np.array(boot_loess)
        conf_interval = st.t.interval(0.95, len(boot_loess) - 1, loc=yout, scale=st.sem(boot_loess))

        ## Find values for HR
        if i == 0:
            print(feat_name)
            x_grid_rounded = [round(xs) for xs in x_grid] # Round the grid to search for values
            if feat_name == 'IL-6':
                print(x_grid_rounded)
            if feat_name == '#Days in Hosp. before ICU':
                print(x_grid_rounded)
            if not math.isnan(hr_ci.loc['Ref', feature_name]):
                if isinstance(hr_ci.loc['Ref', feature_name], int) or isinstance(hr_ci.loc['Ref', feature_name], float):
                    if hr_ci.loc['Ref', feature_name] != -1:
                        hr_ci.loc['HR_ref', feature_name] = yout[x_grid_rounded.index(hr_ci.loc['Ref', feature_name])]
                        hr_ci.loc['HR_step', feature_name] = yout[x_grid_rounded.index(hr_ci.loc['Step', feature_name])]
                        hr_ci.loc['CI_ref_low', feature_name] = conf_interval[0][
                            x_grid_rounded.index(hr_ci.loc['Ref', feature_name])]
                        hr_ci.loc['CI_ref_high', feature_name] = conf_interval[1][
                            x_grid_rounded.index(hr_ci.loc['Ref', feature_name])]
                        hr_ci.loc['CI_step_low', feature_name] = conf_interval[0][
                            x_grid_rounded.index(hr_ci.loc['Step', feature_name])]
                        hr_ci.loc['CI_step_high', feature_name] = conf_interval[1][
                            x_grid_rounded.index(hr_ci.loc['Step', feature_name])]
                        print(hr_ci)

        # Scatter plots with LOESS
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='k', linestyle='dashed')
        sc = plt.scatter(feature_values[feature_name], shap_values[feature_name],
                         cmap=colormaps['coolwarm'], c=col_data, s=40, label=f"SHAP value for {feat_name}")
        ax.plot(x_grid, yout, color='k', label=f"LOESS for SHAP for {feat_name}")
        ax.fill_between(x_grid, conf_interval[0], conf_interval[1], color='k', alpha=0.08)
        # Labels and colorbar
        plt.xlabel(feat_name)
        # Limits for x and y in plot - and add markings of reference values for plots in article
        if feat_name == 'Cystatin C':
            plt.xlim(right=6)
        if feat_name == 'Neutrophils':
            plt.xlim(right=35)
            plt.ylim(-1.5, 3)
        if feat_name == '#Days in Hosp. before ICU':
            plt.xlim(right=32)
            plt.ylim(-4, 4)
        if feat_name == 'IL-6':
            ax.set_xlim(right=4914)
            ax.set_ylim(-4, 4)

            # Draw a round circle in display space, even with unequal axes
            circle_size_pts = 400  # Size in points

            da = DrawingArea(2 * circle_size_pts, 2 * circle_size_pts, 0, 0)

            # x1 circle
            x1idx = [i for i, x in enumerate(x_grid_rounded) if x == 6]  # idx for x1
            if len(x1idx) > 0:
                circle1 = plt.Circle((circle_size_pts, circle_size_pts),
                                     radius=circle_size_pts / 30,
                                     facecolor='none', edgecolor='red', linewidth=2)
                da.add_artist(circle1)
                ab1 = AnnotationBbox(da, (10, yout[x1idx[0]]),
                                     xycoords='data',
                                     frameon=False,
                                     box_alignment=(0.5, 0.5))  # Center the circle

                ax.add_artist(ab1)

            # x2 circle
            x2idx = [i for i, x in enumerate(x_grid_rounded) if x == 1106]  # idx for x2
            if len(x2idx) > 0:
                circle2 = plt.Circle((circle_size_pts, circle_size_pts),
                                     radius=circle_size_pts / 30,
                                     facecolor='none', edgecolor='red', linewidth=2)
                da.add_artist(circle2)
                ab2 = AnnotationBbox(da, (1000, yout[x2idx[0]]),
                                     xycoords='data',
                                     frameon=False,
                                     box_alignment=(0.5, 0.5))  # Center the circle
                ax.add_artist(ab2)

        if feat_name == '#Meds/Patient':
            # Draw a round circle in display space, even with unequal axes
            circle_size_pts = 400  # Size in points

            da = DrawingArea(2*circle_size_pts, 2*circle_size_pts, 0, 0)

            # x1 circle
            x1idx = [i for i, x in enumerate(x_grid_rounded) if x == 2] # idx for x1
            if len(x1idx) > 0:
                circle1 = plt.Circle((circle_size_pts, circle_size_pts),
                                        radius=circle_size_pts / 30,#72,  # Convert points to inches
                                     facecolor='none', edgecolor='red', linewidth=2)
                da.add_artist(circle1)
                ab1 = AnnotationBbox(da, (2, yout[x1idx[2]]),
                                    xycoords='data',
                                    frameon=False,
                                    box_alignment=(0.5, 0.5))  # Center the circle

                ax.add_artist(ab1)

            # x2 circle
            x2idx = [i for i, x in enumerate(x_grid_rounded) if x == 5]  # idx for x2
            if len(x2idx) > 0:
                circle2 = plt.Circle((circle_size_pts, circle_size_pts),
                                        radius=circle_size_pts / 30,#72,  # Convert points to inches
                                     facecolor='none', edgecolor='red', linewidth=2)
                da.add_artist(circle2)
                ab2 = AnnotationBbox(da, (5, yout[x2idx[3]]),
                                    xycoords='data',
                                    frameon=False,
                                    box_alignment=(0.5, 0.5))  # Center the circle
                ax.add_artist(ab2)

        if feat_name == 'CCI':
            plt.xlim(right=10)
            plt.ylim(-2, 4)

        plt.ylabel("SHAP value")
        plt.title(
            "Variable effect on SHAP value for one-year mortality risk"
        )
        cbar = plt.colorbar(sc)
        cbar.set_label(feat_names[i])
        ax.grid(True)
        plt.tight_layout()
        save_as_svg(date_time, 'survival_dep_' + str(feature_name) +
                    '_' + str(coloring_feature), subfolder='dependency_plots')

    return hr_ci

# -------------------------------------------------------------------
# The computed SHAP values
# -------------------------------------------------------------------
def get_shap_values(
        estimator: Any,
        test_x: pd.DataFrame | np.ndarray | torch.Tensor,
        attribute_names: list[str] | None = None,
        **kwargs: Any,
) -> np.ndarray:
    """Compute SHAP values for a model's predictions on input features.

    This function calculates SHAP (SHapley Additive exPlanations) values that
    attribute the contribution of each input feature to the model's output.
    It automatically selects the appropriate SHAP explainer based on the model.

    Args:
        estimator: The model to explain, typically a TabPFNClassifier or scikit-learn compatible model.
        test_x: The input features to compute SHAP values for.
        attribute_names: Column names for the features when test_x is a numpy array.
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        np.ndarray: The computed SHAP values with shape (n_samples, n_features).
    """
    if isinstance(test_x, torch.Tensor):
        test_x = test_x.cpu().numpy()

    if isinstance(test_x, np.ndarray):
        test_x = pd.DataFrame(test_x)
        if attribute_names is not None:
            test_x.columns = attribute_names
        else:
            test_x = test_x.rename(columns={c: str(c) for c in test_x.columns})

    if hasattr(estimator, "predict_function_for_shap"):
        predict_function_for_shap = estimator.predict_function_for_shap
    else:
        predict_function_for_shap = (
            "predict_proba" if hasattr(estimator, "predict_proba") else "predict"
        )

    if hasattr(estimator, "fit_at_predict_time"):
        if not estimator.fit_at_predict_time:
            pass

    def get_shap() -> np.ndarray:
        explainer = get_default_explainer(
            estimator,
            test_x,
            predict_function_for_shap,
            **kwargs,
        )
        return explainer(test_x)

    if hasattr(estimator, "show_progress"):
        show_progress_ = estimator.show_progress
        estimator.show_progress = False
        try:
            shap_values = get_shap()
        finally:
            estimator.show_progress = show_progress_
    else:
        shap_values = get_shap()

    return shap_values

# -------------------------------------------------------------------
# A configured SHAP explainer for the model
# -------------------------------------------------------------------
def get_default_explainer(
        estimator: Any,
        test_x: pd.DataFrame,
        predict_function_for_shap: str | Callable = "predict",
        **kwargs: Any,
) -> Any:
    """Create a standard SHAP explainer for non-TabPFN models.

    Args:
        estimator: The model to explain.
        test_x: The input features to compute SHAP values for.
        predict_function_for_shap: Function name or callable to use for prediction.
            Defaults to "predict".
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        Any: A configured SHAP explainer for the model.
    """
    import shap

    shap.maskers.Independent(test_x, max_samples=1000)

    return shap.Explainer(
        getattr(estimator, predict_function_for_shap)
        if isinstance(predict_function_for_shap, str)
        else predict_function_for_shap,
        test_x,
        **kwargs,
    )


# -------------------------------------------------------------------
# Calculate SHAP hazard ratios with 95% confidence intervals
# -------------------------------------------------------------------
def shap_hr(shap_all, feature_all, topfeatures, just_get_df = False):
    """
    Calculate hazard ratios (HR) and confidence intervals (CI) from SHAP values
    for binary, ordinal, and continuous variables via bootstrapping.

    This function uses SHAP values and feature data to estimate the relative
    impact of different variables on model predictions in terms of hazard ratios
    (similar to Cox regression outputs). It computes HRs and 95% confidence
    intervals for each feature type by repeatedly resampling (bootstrapping)
    the dataset and comparing SHAP-derived effects.

    Parameters
    ----------
    shap_all : pandas.DataFrame
        A DataFrame containing SHAP values for all samples and features.
        Each column corresponds to a feature, and each row to an observation.

    feature_all : pandas.DataFrame
        The corresponding feature values for the same samples and features
        as in `shap_all`.

    topfeatures : list of str
        List of top-ranked feature names (based on importance) to include in
        hazard ratio estimation.

    just_get_df : bool, optional (default=False)
        If True, returns only the manually defined reference/step dataframe
        (without computing HR and CI statistics).

    Returns
    -------
    hr_ci_dataframe : pandas.DataFrame
        A DataFrame with manually defined reference (Ref) and step (Step)
        values for each top feature, along with placeholders for HR and CI
        results.

    Side Effects
    -------------
    - Creates and saves several Excel files (via `excel()` function) containing
      bootstrapped hazard ratio statistics for:
        * Binary variables (e.g., medication, comorbidity indicators)
        * Ordinal variables (e.g., smoking status, noradrenaline use, CCI)
        * Continuous variables (e.g., IL-6, neutrophils, BMI)
    - Uses bootstrapping (controlled by global variables such as
      `n_bootstraps`, `rng`, `size_bootstrap`, and `REPETITIONS`).
    - Uses `loess_1d` smoothing for continuous SHAP effects.

    Method Overview
    ---------------
    1. **Setup reference points**:
       Manually assigns a reference (Ref) and comparison step (Step) for each
       continuous variable based on clinically meaningful thresholds.

    2. **Binary variable analysis**:
       - Computes mean SHAP values for feature==0 and feature==1.
       - Calculates log hazard ratios as differences between these means.
       - Bootstraps across subsets of data to estimate variability and
         confidence intervals.

    3. **Ordinal variable analysis**:
       - Performs pairwise comparisons between ordered categorical levels
         (e.g., smoker_0 vs smoker_1 vs smoker_2).
       - Uses bootstrapping to estimate mean SHAP differences and derive HRs
         with confidence intervals.

    4. **Continuous variable analysis**:
       - Fits LOESS (locally weighted regression) curves to SHAP vs feature
         relationships.
       - Estimates log(HR) as the SHAP difference between “Step” and “Ref”
         feature values.
       - Bootstraps results to estimate uncertainty and compute 95% CIs.

    5. **Output and saving**:
       - Results are saved into separate Excel sheets for binary, ordinal, and
         continuous features (names prefixed with `_shap_*_stats`).
       - Each output includes hazard ratio, log(HR), confidence intervals,
         z-scores, and p-values (both raw and rounded).

    Notes
    -----
    - This function relies on several external/global variables and functions:
        * `rng`, `n_bootstraps`, `size_bootstrap`, `REPETITIONS`
        * `loess_1d` for smoothing
        * `excel()` for result export
        * `change_names` and `NAME` for column renaming and output naming
    - The HRs are derived from SHAP-based differences, not a formal
      time-to-event regression model, so interpretation should consider
      this methodological difference.

    """
    # Dataframe to store HR values (Manually chosen values by Attila)
    hr_ci_dataframe = pd.DataFrame([], columns=topfeatures,
                                   index=['Ref', 'Step', 'HR_ref', 'CI_ref_low', 'CI_ref_high', 'HR_step',
                                          'CI_step_low', 'CI_step_high'])
    for name in topfeatures:
        if name == 'age_at_admission':
            hr_ci_dataframe.loc['Ref', name] = 60
            hr_ci_dataframe.loc['Step', name] = 75
        if name == 'cystatin_c':
            hr_ci_dataframe.loc['Ref', name] = 1
            hr_ci_dataframe.loc['Step', name] = 3
        if name == 'leukocytes':
            hr_ci_dataframe.loc['Ref', name] = 5
            hr_ci_dataframe.loc['Step', name] = 20
        if name == 'neutrofila':
            hr_ci_dataframe.loc['Ref', name] = 5
            hr_ci_dataframe.loc['Step', name] = 20
        if name == 'albumin':
            hr_ci_dataframe.loc['Ref', name] = 22
            hr_ci_dataframe.loc['Step', name] = 32
        if name == 'clinical_frailty_scale':
            hr_ci_dataframe.loc['Ref', name] = 2
            hr_ci_dataframe.loc['Step', name] = 4
        if name == 'CCI':
            hr_ci_dataframe.loc['Ref', name] = -1  # 3
            hr_ci_dataframe.loc['Step', name] = -1  # 6
        if name == 'meds_per_pat':
            hr_ci_dataframe.loc['Ref', name] = 2
            hr_ci_dataframe.loc['Step', name] = 5
        if name == 'il6':
            hr_ci_dataframe.loc['Ref', name] = 6  # closest value to 54
            hr_ci_dataframe.loc['Step', name] = 1106  # closest value to 1096
        if name == 'time_at_hospital_before_admission':
            hr_ci_dataframe.loc['Ref', name] = 4
            hr_ci_dataframe.loc['Step', name] = 11
        if name == 'BMI':
            hr_ci_dataframe.loc['Ref', name] = 25
            hr_ci_dataframe.loc['Step', name] = 35

    # Just return the dataframe to use in other function
    if just_get_df:
        return hr_ci_dataframe

    # Calculate HR and CI
    else:

        ### HR SHAP for binary variables ###
        stats_bin = {}
        # Get binary variables
        bin_var = ['C03CA', 'A10BK', 'N06AX', 'N05BA', 'N03AX', 'B01AF', 'C01DA', 'N06AB', 'B03BA', 'N02AA', 'N05CF',
                   'R03AC', 'R03AK', 'H02AB', 'A10BJ', 'C10AA', 'B01AC', 'C07AB', 'A02BC', 'C08CA', 'A12AX', 'N02BE',
                   'B03BB', 'A10BA', 'Y92BA', 'Y92AD', 'H03AA', 'M04AA', 'congestive_heart_failure', 'cancer', 'NOAK',
                   'hypertension', 'myocardial_infarction', 'peripheral_vascular_disease', 'cerebrovascular_insult',
                   'chronic_pulmonary_disease', 'rheumatic_disease', 'peptic_ulcer_disease', 'diabetes_mellitus_uncompl',
                   'diabetes_mellitus_compl', 'chronic_kidney_disease', 'imv_within_24h', 'sex_male', 'no_CPR', 'ACEi',
                   'ARB']

        all_indices = shap_all.index.to_list()
        for ii in range(n_bootstraps):
            indices = rng.choice(all_indices,
                                 size=int(len(shap_all) / size_bootstrap),
                                 replace=False)  # to not get two of the same value that leads to error in division
            all_indices = [e for e in all_indices if e not in indices.tolist()]
            stats_bin[ii] = pd.DataFrame([], index=['mean_pos/pos_std', 'mean_neg/neg_std', 'log_hr'],
                                         columns=bin_var)

            for var in bin_var:
                # Get position of 0
                zero_idx = feature_all.loc[indices, var].loc[feature_all[var] == 0].index
                # Get position of 1
                one_idx = feature_all.loc[indices, var].loc[feature_all[var] == 1].index

                neg = np.mean(shap_all.loc[indices, var].loc[
                                  shap_all.loc[indices, var].index.isin(zero_idx)])
                neg_std = np.std(shap_all.loc[indices, var].loc[
                                     shap_all.loc[indices, var].index.isin(zero_idx)])
                pos = np.mean(shap_all.loc[indices, var].loc[
                                  shap_all.loc[indices, var].index.isin(one_idx)])
                pos_std = np.std(shap_all.loc[indices, var].loc[
                                     shap_all.loc[indices, var].index.isin(one_idx)])

                # Dispersion taken into regard
                if neg_std == 0 and pos_std != 0:
                    ave = (pos / pos_std - neg)
                    stats_bin[ii].loc['mean_neg/neg_std', var] = neg
                    stats_bin[ii].loc['mean_pos/pos_std', var] = pos / pos_std
                elif neg_std != 0 and pos_std == 0:
                    ave = (pos - neg / neg_std)
                    stats_bin[ii].loc['mean_neg/neg_std', var] = neg / pos_std
                    stats_bin[ii].loc['mean_pos/pos_std', var] = pos
                elif neg_std == 0 and pos_std == 0:
                    ave = (pos - neg)
                    stats_bin[ii].loc['mean_neg/neg_std', var] = neg
                    stats_bin[ii].loc['mean_pos/pos_std', var] = pos
                else:
                    ave = (pos / pos_std - neg / neg_std)
                    stats_bin[ii].loc['mean_neg/neg_std', var] = neg / pos_std
                    stats_bin[ii].loc['mean_pos/pos_std', var] = pos / pos_std

                # Add to df
                stats_bin[ii].loc['log_hr', var] = ave

        # Calculate std and ci:s given means
        bin_mean = pd.DataFrame(columns=bin_var, index=range(n_bootstraps))

        # SHAP values
        for n, df in stats_bin.items():
            bin_mean.iloc[n] = df.loc['log_hr'].values

        # Add stats to df
        shap_bin_stats = pd.DataFrame(columns=bin_var)
        for variable_bin in bin_var:
            # The bootstrapped SHAP mean and std
            aver = np.mean(bin_mean[variable_bin])
            std = np.std(bin_mean[variable_bin])
            shap_bin_stats.loc['log_hr', variable_bin] = aver
            shap_bin_stats.loc['std', variable_bin] = std
            shap_bin_stats.loc['z', variable_bin] = aver / std  # Andreas says OK!
            shap_bin_stats.loc['p-value', variable_bin] = 2 * (
                        1 - st.norm.cdf(np.abs(shap_bin_stats.loc['z', variable_bin])))  # Andreas says OK!
            shap_bin_stats.loc['hazard_ratio', variable_bin] = np.exp(aver)

            # CI
            ci = st.t.interval(0.95, len(bin_mean[variable_bin]) - 1, loc=aver,
                               scale=std)

            # The bootstrapped 95% CI
            shap_bin_stats.loc['ci_low_lg', variable_bin] = ci[0]
            shap_bin_stats.loc['ci_high_lg', variable_bin] = ci[1]
            shap_bin_stats.loc['ci_low', variable_bin] = np.exp(ci[0])
            shap_bin_stats.loc['ci_high', variable_bin] = np.exp(ci[1])

            # Rounded numbers
            try:
                # HR
                shap_bin_stats.loc['hazard_ratio_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['hazard_ratio', variable_bin]).quantize(Decimal('0.01'),
                                                                                       rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI low
                shap_bin_stats.loc['ci_low_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['ci_low', variable_bin]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI high
                shap_bin_stats.loc['ci_high_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['ci_high', variable_bin]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log HR
                shap_bin_stats.loc['log_hr_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['log_hr', variable_bin]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI low
                shap_bin_stats.loc['ci_low_lg_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['ci_low_lg', variable_bin]).quantize(Decimal('0.01'),
                                                                                    rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI high
                shap_bin_stats.loc['ci_high_lg_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['ci_high_lg', variable_bin]).quantize(Decimal('0.01'),
                                                                                     rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # P-value
                shap_bin_stats.loc['p_value_rounded', variable_bin] = float(
                    Decimal(shap_bin_stats.loc['p-value', variable_bin]).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue

            # Number variable_bin == 1
            shap_bin_stats.loc['n=1', variable_bin] = (feature_all[variable_bin] == 1).sum() / REPETITIONS
            # Number variable_bin == 0
            shap_bin_stats.loc['n=0', variable_bin] = (feature_all[variable_bin] == 0).sum() / REPETITIONS

        # Change the variable name of the dataframe
        shap_bin_stats.rename(columns=change_names, inplace=True)

        # Save df for binary variables
        excel(date_time, shap_bin_stats, str(NAME) + '_shap_bin_stats', subfolder='SHAP')

        ### HR SHAP for ordinal variables ###
        stats_ord_smoker = {}
        stats_ord_noradrenaline = {}
        stats_ord_cci = {}

        smoker = ['smoker_01', 'smoker_12', 'smoker_02']
        noradre = ['noradrenaline_01', 'noradrenaline_12', 'noradrenaline_02']
        ccis = ['cci_01', 'cci_12', 'cci_23', 'cci_34', 'cci_45', 'cci_56', 'cci_67', 'cci_78', 'cci_89', 'cci_910',
                'cci_1011']  # only full-case


        # Bootstrap
        all_indices = shap_all.index.to_list()
        for ii in range(n_bootstraps):
            indices = rng.choice(all_indices,
                                 size=int(len(shap_all) / size_bootstrap),
                                 replace=False)  # to not get two of the same value that leads to error in division
            all_indices = [e for e in all_indices if e not in indices.tolist()]

            # Smoker
            stats_ord_smoker[ii] = pd.DataFrame()
            means = []
            stds = []
            for val in range(len(feature_all['smoker'].unique())):
                idx = feature_all.loc[indices, 'smoker'].loc[feature_all['smoker'] == val].index
                means.append(np.mean(shap_all.loc[indices, 'smoker'].loc[
                                         shap_all.loc[indices, 'smoker'].index.isin(idx)]))
                stds.append(np.std(shap_all.loc[indices, 'smoker'].loc[
                                       shap_all.loc[indices, 'smoker'].index.isin(idx)]))
            if stds[0] == 0 or np.isnan(stds[0]):
                if stds[1] == 0 or np.isnan(stds[1]):
                    stats_ord_smoker[ii].loc['mean', 'smoker_01'] = means[1] - means[0]
                elif stds[1] != 0:
                    stats_ord_smoker[ii].loc['mean', 'smoker_01'] = means[1] / stds[1] - means[0]
                if len(stds) == 3:
                    if stds[2] == 0 or np.isnan(stds[2]):
                        stats_ord_smoker[ii].loc['mean', 'smoker_02'] = means[2] - means[0]
                    elif stds[2] != 0:
                        stats_ord_smoker[ii].loc['mean', 'smoker_02'] = means[2] / stds[2] - means[0]
            elif stds[0] != 0:
                if stds[1] == 0 or np.isnan(stds[1]):
                    stats_ord_smoker[ii].loc['mean', 'smoker_01'] = means[1] - means[0] / stds[0]
                elif stds[1] != 0:
                    stats_ord_smoker[ii].loc['mean', 'smoker_01'] = means[1] / stds[1] - means[0] / stds[0]
                if len(stds) == 3:
                    if stds[2] == 0 or np.isnan(stds[2]):
                        stats_ord_smoker[ii].loc['mean', 'smoker_02'] = means[2] - means[0] / stds[0]
                    elif stds[2] != 0:
                        stats_ord_smoker[ii].loc['mean', 'smoker_02'] = means[2] / stds[2] - means[0] / stds[0]
            if len(stds) == 3:
                if stds[1] == 0 or np.isnan(stds[1]):
                    if stds[2] == 0 or np.isnan(stds[2]):
                        stats_ord_smoker[ii].loc['mean', 'smoker_12'] = means[2] - means[1]
                    elif stds[2] != 0:
                        stats_ord_smoker[ii].loc['mean', 'smoker_12'] = means[2] / stds[2] - means[1]
                elif stds[1] != 0:
                    if stds[2] == 0 or np.isnan(stds[2]):
                        stats_ord_smoker[ii].loc['mean', 'smoker_12'] = means[2] - means[1] / stds[1]
                    elif stds[2] != 0:
                        stats_ord_smoker[ii].loc['mean', 'smoker_12'] = means[2] / stds[2] - means[1] / stds[1]

            # Noradrenaline
            stats_ord_noradrenaline[ii] = pd.DataFrame()
            means = []
            stds = []
            for val in range(len(feature_all['noradrenaline'].unique())):
                idx = feature_all.loc[indices, 'noradrenaline'].loc[
                    feature_all['noradrenaline'] == val].index
                means.append(np.mean(
                    shap_all.loc[indices, 'noradrenaline'].loc[
                        shap_all.loc[indices, 'noradrenaline'].index.isin(idx)]))
                stds.append(np.std(
                    shap_all.loc[indices, 'noradrenaline'].loc[
                        shap_all.loc[indices, 'noradrenaline'].index.isin(idx)]))

            if stds[0] == 0 or np.isnan(stds[0]):
                if stds[1] == 0 or np.isnan(stds[1]):
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_01'] = means[1] - means[0]
                elif stds[1] != 0:
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_01'] = means[1] / stds[1] - means[0]
                if stds[2] == 0 or np.isnan(stds[2]):
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_02'] = means[2] - means[0]
                elif stds[2] != 0:
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_02'] = means[2] / stds[2] - means[0]
            elif stds[0] != 0:
                if stds[1] == 0 or np.isnan(stds[1]):
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_01'] = means[1] - means[0] / stds[0]
                elif stds[1] != 0:
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_01'] = means[1] / stds[1] - means[0] / stds[0]
                if stds[2] == 0 or np.isnan(stds[2]):
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_02'] = means[2] - means[0] / stds[0]
                elif stds[2] != 0:
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_02'] = means[2] / stds[2] - means[0] / stds[0]
            if stds[1] == 0 or np.isnan(stds[1]):
                if stds[2] == 0 or np.isnan(stds[2]):
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_12'] = means[2] - means[1]
                elif stds[2] != 0:
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_12'] = means[2] / stds[2] - means[1]
            elif stds[1] != 0:
                if stds[2] == 0 or np.isnan(stds[2]):
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_12'] = means[2] - means[1] / stds[1]
                elif stds[2] != 0:
                    stats_ord_noradrenaline[ii].loc['mean', 'noradrenaline_12'] = means[2] / stds[2] - means[1] / stds[1]

            # CCI
            stats_ord_cci[ii] = pd.DataFrame()
            means = []
            stds = []
            cci_values = feature_all['CCI'].unique()
            # Nans are added if there are not any of that number in the sample. Remove NaNs later when calculating HR and CI
            for val in sorted(cci_values.tolist()):
                idx = feature_all.loc[indices, 'CCI'].loc[feature_all['CCI'] == val].index
                means.append(
                    np.mean(shap_all.loc[indices, 'CCI'].loc[
                                shap_all.loc[indices, 'CCI'].index.isin(idx)]))
                stds.append(
                    np.std(shap_all.loc[indices, 'CCI'].loc[
                               shap_all.loc[indices, 'CCI'].index.isin(idx)]))

            for i in range(11):
                lower = i
                upper = i + 1
                name = 'cci_' + str(lower) + str(upper)
                if stds[lower] == 0 or np.isnan(stds[lower]):
                    if stds[upper] == 0 or np.isnan(stds[upper]):
                        stats_ord_cci[ii].loc['mean', name] = means[upper] - means[lower]
                    elif stds[upper] != 0:
                        stats_ord_cci[ii].loc['mean', name] = means[upper] / stds[upper] - means[lower]
                elif stds[lower] != 0:
                    if stds[upper] == 0 or np.isnan(stds[upper]):
                        stats_ord_cci[ii].loc['mean', name] = means[upper] - means[lower] / stds[lower]
                    elif stds[upper] != 0:
                        stats_ord_cci[ii].loc['mean', name] = means[upper] / stds[upper] - means[lower] / stds[lower]

            # Add comparison of CCI 3 contra CCI 6
            lower = 3
            upper = 6
            name = 'cci_' + str(lower) + str(upper)
            if stds[lower] == 0 or np.isnan(stds[lower]):
                if stds[upper] == 0 or np.isnan(stds[upper]):
                    stats_ord_cci[ii].loc['mean', name] = means[upper] - means[lower]
                elif stds[upper] != 0:
                    stats_ord_cci[ii].loc['mean', name] = means[upper] / stds[upper] - means[lower]
            elif stds[lower] != 0:
                if stds[upper] == 0 or np.isnan(stds[upper]):
                    stats_ord_cci[ii].loc['mean', name] = means[upper] - means[lower] / stds[lower]
                elif stds[upper] != 0:
                    stats_ord_cci[ii].loc['mean', name] = means[upper] / stds[upper] - means[lower] / stds[lower]

        ### Calculate std and ci:s given means
        # Smoker
        smoker_mean = pd.DataFrame(columns=smoker, index=range(n_bootstraps))
        # SHAP values
        for n, df in stats_ord_smoker.items():
            smoker_mean.iloc[n] = df.loc['mean'].values

        # Noradrenaline
        noradrenaline_mean = pd.DataFrame(columns=noradre, index=range(n_bootstraps))
        # SHAP values
        for n, df in stats_ord_noradrenaline.items():
            noradrenaline_mean.iloc[n] = df.loc['mean'].values

        # CCI
        print(ccis)
        ccis.append('cci_36')  # Extend with the comparison based on clinical reference points
        print(ccis)
        cci_mean = pd.DataFrame(columns=ccis, index=range(n_bootstraps))
        # SHAP values
        for n, df in stats_ord_cci.items():
            cci_mean.iloc[n] = df.loc['mean'].values

        # Add stats to df
        shap_smoker_stats = pd.DataFrame(columns=smoker)
        for idxx, variable_ord in enumerate(smoker):
            print(variable_ord)
            non_na_means = smoker_mean[variable_ord].dropna()
            # The bootstrapped SHAP mean and std
            aver = np.mean(non_na_means)
            std = np.std(non_na_means)
            shap_smoker_stats.loc['log_hr', variable_ord] = aver
            shap_smoker_stats.loc['std', variable_ord] = std
            shap_smoker_stats.loc['z', variable_ord] = aver / std
            shap_smoker_stats.loc['p-value', variable_ord] = 2 * (
                        1 - st.norm.cdf(np.abs(shap_smoker_stats.loc['z', variable_ord])))
            shap_smoker_stats.loc['hazard_ratio', variable_ord] = np.exp(aver)

            # CI
            ci = st.t.interval(0.95, len(non_na_means) - 1, loc=aver,
                               scale=std)

            # The bootstrapped 95% CI
            shap_smoker_stats.loc['ci_low_lg', variable_ord] = ci[0]
            shap_smoker_stats.loc['ci_high_lg', variable_ord] = ci[1]
            shap_smoker_stats.loc['ci_low', variable_ord] = np.exp(ci[0])
            shap_smoker_stats.loc['ci_high', variable_ord] = np.exp(ci[1])

            # Rounded numbers
            try:
                # HR
                shap_smoker_stats.loc['hazard_ratio_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['hazard_ratio', variable_ord]).quantize(Decimal('0.01'),
                                                                                          rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI low
                shap_smoker_stats.loc['ci_low_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['ci_low', variable_ord]).quantize(Decimal('0.01'),
                                                                                    rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI high
                shap_smoker_stats.loc['ci_high_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['ci_high', variable_ord]).quantize(Decimal('0.01'),
                                                                                     rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log HR
                shap_smoker_stats.loc['log_hr_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['log_hr', variable_ord]).quantize(Decimal('0.01'),
                                                                                    rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI low
                shap_smoker_stats.loc['ci_low_lg_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['ci_low_lg', variable_ord]).quantize(Decimal('0.01'),
                                                                                       rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI high
                shap_smoker_stats.loc['ci_high_lg_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['ci_high_lg', variable_ord]).quantize(Decimal('0.01'),
                                                                                        rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # P-value
                shap_smoker_stats.loc['p_value_rounded', variable_ord] = float(
                    Decimal(shap_smoker_stats.loc['p-value', variable_ord]).quantize(Decimal('0.001'),
                                                                                     rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue

            # Number variable_ord
            shap_smoker_stats.loc['n', variable_ord] = (feature_all[
                                                            'smoker'] == idxx).sum() / REPETITIONS  # ['smoker_01', 'smoker_12', 'smoker_02']

        # Change the variable name of the dataframe
        shap_smoker_stats.rename(columns=change_names, inplace=True)

        # Save df for binary variables
        excel(date_time, shap_smoker_stats, str(NAME) + '_shap_smoker_stats', subfolder='SHAP')

        shap_noradrenaline_stats = pd.DataFrame(columns=noradre)
        for idxx, variable_ord in enumerate(noradre):
            print(variable_ord)
            non_na_means = noradrenaline_mean[variable_ord].dropna()
            # The bootstrapped SHAP mean and std
            aver = np.mean(non_na_means)
            std = np.std(non_na_means)
            shap_noradrenaline_stats.loc['log_hr', variable_ord] = aver
            shap_noradrenaline_stats.loc['std', variable_ord] = std
            shap_noradrenaline_stats.loc['z', variable_ord] = aver / std
            shap_noradrenaline_stats.loc['p-value', variable_ord] = 2 * (
                        1 - st.norm.cdf(np.abs(shap_noradrenaline_stats.loc['z', variable_ord])))
            shap_noradrenaline_stats.loc['hazard_ratio', variable_ord] = np.exp(aver)

            # CI
            ci = st.t.interval(0.95, len(non_na_means) - 1, loc=aver,
                               scale=std)

            # The bootstrapped 95% CI
            shap_noradrenaline_stats.loc['ci_low_lg', variable_ord] = ci[0]
            shap_noradrenaline_stats.loc['ci_high_lg', variable_ord] = ci[1]
            shap_noradrenaline_stats.loc['ci_low', variable_ord] = np.exp(ci[0])
            shap_noradrenaline_stats.loc['ci_high', variable_ord] = np.exp(ci[1])

            # Rounded numbers
            try:
                # HR
                shap_noradrenaline_stats.loc['hazard_ratio_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['hazard_ratio', variable_ord]).quantize(Decimal('0.01'),
                                                                                                 rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI low
                shap_noradrenaline_stats.loc['ci_low_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['ci_low', variable_ord]).quantize(Decimal('0.01'),
                                                                                           rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI high
                shap_noradrenaline_stats.loc['ci_high_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['ci_high', variable_ord]).quantize(Decimal('0.01'),
                                                                                            rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log HR
                shap_noradrenaline_stats.loc['log_hr_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['log_hr', variable_ord]).quantize(Decimal('0.01'),
                                                                                           rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI low
                shap_noradrenaline_stats.loc['ci_low_lg_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['ci_low_lg', variable_ord]).quantize(Decimal('0.01'),
                                                                                              rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI high
                shap_noradrenaline_stats.loc['ci_high_lg_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['ci_high_lg', variable_ord]).quantize(Decimal('0.01'),
                                                                                               rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # P-value
                shap_noradrenaline_stats.loc['p_value_rounded', variable_ord] = float(
                    Decimal(shap_noradrenaline_stats.loc['p-value', variable_ord]).quantize(Decimal('0.001'),
                                                                                            rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue

            # Number variable_ord
            shap_noradrenaline_stats.loc['n', variable_ord] = (feature_all[
                                                                   'noradrenaline'] == idxx).sum() / REPETITIONS

        # Change the variable name of the dataframe
        shap_noradrenaline_stats.rename(columns=change_names, inplace=True)

        # Save df for binary variables
        excel(date_time, shap_noradrenaline_stats, str(NAME) + '_shap_noradrenaline_stats', subfolder='SHAP')

        print(ccis)
        shap_cci_stats = pd.DataFrame(columns=ccis)
        for idxx, variable_ord in enumerate(ccis):
            print(variable_ord)
            non_na_means = cci_mean[variable_ord].dropna()
            # The bootstrapped SHAP mean and std
            aver = np.mean(non_na_means)
            std = np.std(non_na_means)
            shap_cci_stats.loc['log_hr', variable_ord] = aver
            shap_cci_stats.loc['std', variable_ord] = std
            shap_cci_stats.loc['z', variable_ord] = aver / std
            shap_cci_stats.loc['p-value', variable_ord] = 2 * (
                        1 - st.norm.cdf(np.abs(shap_cci_stats.loc['z', variable_ord])))
            shap_cci_stats.loc['hazard_ratio', variable_ord] = np.exp(aver)

            # CI
            ci = st.t.interval(0.95, len(non_na_means) - 1, loc=aver,
                               scale=std)

            # The bootstrapped 95% CI
            shap_cci_stats.loc['ci_low_lg', variable_ord] = ci[0]
            shap_cci_stats.loc['ci_high_lg', variable_ord] = ci[1]
            shap_cci_stats.loc['ci_low', variable_ord] = np.exp(ci[0])
            shap_cci_stats.loc['ci_high', variable_ord] = np.exp(ci[1])

            # Rounded numbers
            try:
                # HR
                shap_cci_stats.loc['hazard_ratio_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['hazard_ratio', variable_ord]).quantize(Decimal('0.01'),
                                                                                       rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI low
                shap_cci_stats.loc['ci_low_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['ci_low', variable_ord]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # CI high
                shap_cci_stats.loc['ci_high_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['ci_high', variable_ord]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log HR
                shap_cci_stats.loc['log_hr_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['log_hr', variable_ord]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI low
                shap_cci_stats.loc['ci_low_lg_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['ci_low_lg', variable_ord]).quantize(Decimal('0.01'),
                                                                                    rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # Log CI high
                shap_cci_stats.loc['ci_high_lg_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['ci_high_lg', variable_ord]).quantize(Decimal('0.01'),
                                                                                     rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue
            try:
                # P-value
                shap_cci_stats.loc['p_value_rounded', variable_ord] = float(
                    Decimal(shap_cci_stats.loc['p-value', variable_ord]).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
            except InvalidOperation:
                continue


            # Number variable_ord
            shap_cci_stats.loc['n', variable_ord] = (feature_all['CCI'] == idxx).sum() / REPETITIONS

        # Change the variable name of the dataframe
        shap_cci_stats.rename(columns=change_names, inplace=True)

        # Save df for binary variables
        excel(date_time, shap_cci_stats, str(NAME) + '_shap_cci_stats', subfolder='SHAP')

        ### HR SHAP for continuous variables ###
        # -------------------------------------------------------------------
        # Calculates SHAP HR and CI based on bootstrapped LOESS curves
        # -------------------------------------------------------------------
        def shap_stats_cont(variables, features, shaps, hr_ci_df):

            stats_cont = {}

            # Bootstrap for each variable
            all_indices = shap_all.index.to_list()
            for ii in range(n_bootstraps):
                indices = rng.choice(all_indices,
                                     size=int(len(shap_all) / size_bootstrap),
                                     replace=False)  # to not get two of the same value that leads to error in division
                all_indices = [e for e in all_indices if e not in indices.tolist()]

                # Initiate dataframe in dict
                stats_cont[ii] = pd.DataFrame([], index=['mean_step', 'mean_ref', 'log_hr'], columns=variables)

                for var in variables:
                    print(var)
                    if var in hr_ci_df.columns.to_list():
                        # Find indices for the variable
                        bootstrap_features = features.loc[indices, var]
                        bootstrap_shaps = shaps.loc[indices, var]

                        # Convert everything to NumPy arrays
                        raw_feature = np.asarray(bootstrap_features).ravel()
                        raw_shap = np.asarray(bootstrap_shaps).ravel()

                        # Grid for smoothing
                        x_grid = np.linspace(np.min(raw_feature), np.max(raw_feature), len(raw_feature))
                        x_grid = np.asarray(x_grid).ravel()

                        # Apply loess
                        xout, yout, wout = loess_1d.loess_1d(raw_feature, raw_shap,
                                                             x_grid, degree=2, frac=0.6)

                        x_grid_rounded = [round(xs) for xs in x_grid]
                        # Find the value in x_grid_rounded closest to chosen ref or step
                        x_ref = min(x_grid_rounded, key=lambda x: abs(x - hr_ci_df.loc['Ref', var]))
                        x_step = min(x_grid_rounded, key=lambda x: abs(x - hr_ci_df.loc['Step', var]))
                        print(hr_ci_df.loc['Ref', var], '-->', x_ref, '-->', yout[x_grid_rounded.index(x_ref)])
                        print(hr_ci_df.loc['Step', var], '-->', x_step, '-->', yout[x_grid_rounded.index(x_step)])

                        stats_cont[ii].loc['mean_ref', var] = yout[x_grid_rounded.index(x_ref)]
                        stats_cont[ii].loc['mean_step', var] = yout[x_grid_rounded.index(x_step)]
                        stats_cont[ii].loc['log_hr', var] = stats_cont[ii].loc['mean_step', var] - stats_cont[ii].loc[
                            'mean_ref', var]

            # Get CI
            cont_mean = pd.DataFrame(columns=variables, index=range(n_bootstraps))
            # SHAP values
            for n, df in stats_cont.items():
                cont_mean.iloc[n] = df.loc['log_hr'].values

            # Add stats to df
            shap_cont_stats = pd.DataFrame(columns=variables)
            print(cont_mean)
            for variable_cont in variables:
                print(variable_cont)
                # The bootstrapped SHAP mean and std
                aver = np.mean(cont_mean[variable_cont])
                std = np.std(cont_mean[variable_cont])
                shap_cont_stats.loc['log_hr', variable_cont] = aver
                shap_cont_stats.loc['std', variable_cont] = std
                shap_cont_stats.loc['z', variable_cont] = aver / std
                shap_cont_stats.loc['p-value', variable_cont] = 2 * (
                        1 - st.norm.cdf(
                    np.abs(shap_cont_stats.loc['z', variable_cont])))  # assuming Gaussian distribution of log(HR)?
                shap_cont_stats.loc['hazard_ratio', variable_cont] = np.exp(aver)

                # CI
                ci = st.t.interval(0.95, len(cont_mean[variable_cont]) - 1, loc=aver,
                                   scale=std)

                # The bootstrapped 95% CI
                shap_cont_stats.loc['ci_low_lg', variable_cont] = ci[0]
                shap_cont_stats.loc['ci_high_lg', variable_cont] = ci[1]
                shap_cont_stats.loc['ci_low', variable_cont] = np.exp(ci[0])
                shap_cont_stats.loc['ci_high', variable_cont] = np.exp(ci[1])

                # Rounded HR + CI
                shap_cont_stats.loc['hazard_ratio_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['hazard_ratio', variable_cont]).quantize(Decimal('0.01'),
                                                                                         rounding=ROUND_HALF_UP))
                shap_cont_stats.loc['ci_low_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['ci_low', variable_cont]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                shap_cont_stats.loc['ci_high_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['ci_high', variable_cont]).quantize(Decimal('0.01'),
                                                                                    rounding=ROUND_HALF_UP))

                shap_cont_stats.loc['log_hr_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['log_hr', variable_cont]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                shap_cont_stats.loc['ci_low_lg_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['ci_low_lg', variable_cont]).quantize(Decimal('0.01'),
                                                                                      rounding=ROUND_HALF_UP))
                shap_cont_stats.loc['ci_high_lg_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['ci_high_lg', variable_cont]).quantize(Decimal('0.01'),
                                                                                       rounding=ROUND_HALF_UP))

                # Rounded p-value
                shap_cont_stats.loc['p_value_rounded', variable_cont] = float(
                    Decimal(shap_cont_stats.loc['p-value', variable_cont]).quantize(Decimal('0.001'),
                                                                                    rounding=ROUND_HALF_UP))

            # Change the variable name of the dataframe
            shap_cont_stats.rename(columns=change_names, inplace=True)

            # Save df for binary variables
            excel(date_time, shap_cont_stats, str(NAME) + '_shap_cont_stats', subfolder='SHAP')

            # End of function

        # SHAP stats for continuous variables
        top14_con_var = ['il6', 'neutrofila',
                         'age_at_admission', 'time_at_hospital_before_admission',
                         'clinical_frailty_scale', 'ICU_strain', 'albumin',
                         'cystatin_c', 'meds_per_pat', 'CRP', 'BMI', 'leukocytes']

        # Run script for SHAP HR for continuous variables
        shap_stats_cont(top14_con_var, feature_all, shap_all, hr_ci_dataframe)

        return hr_ci_dataframe

# -------------------------------------------------------------------
# Visualize summary plots
# -------------------------------------------------------------------
def fig_summary(shapv, featv, max_disp = 14):
    """
    Generate and save summary visualizations of SHAP values for an XGBoost survival model.

    This function produces two summary plots:
    (1) A customized beeswarm plot for the top `max_disp` most important features, and
    (2) A full SHAP summary plot including all model features.

    These plots help visualize the overall feature importance and the distribution of
    SHAP values (i.e., the contribution of each feature to the model’s predicted risk).

    Parameters
    ----------
    shapv : pandas.DataFrame or numpy.ndarray
        Matrix of SHAP values with shape (n_samples, n_features). Each value indicates
        how much a given feature contributed to the prediction for a single sample.

    featv : pandas.DataFrame
        Corresponding feature values for each sample. The column names should match
        the feature names in `shapv`.

    max_disp : int, optional (default=14)
        The maximum number of top features to display in the first (beeswarm) summary plot.
        The remaining features are grouped together.

    Returns
    -------
    None
        The function saves plots as SVG files using `save_as_svg()` but does not return any value.

    Side Effects
    -------------
    - Saves two figures to disk:
        1. 'XGBoost_survival_summary_plot_<max_disp>.svg' — top features (custom beeswarm)
        2. 'XGBoost_survival_summary_plot_full.svg' — all features (SHAP summary plot)
    - Requires global variable `date_time` for filename tagging.
    - Relies on helper functions:
        * `beeswarm()` — custom SHAP beeswarm plotting function
        * `save_as_svg()` — saves the current matplotlib figure as SVG

    """
    # Top features
    plt.figure()
    beeswarm(np.array(shapv), np.array(featv), featv.columns.to_list(),
             shapv.shape, max_display=max_disp, show=False)
    save_as_svg(date_time, 'XGBoost_survival_summary_plot_' + str(max_disp), subfolder=None)

    # All features
    plt.figure()
    shap.summary_plot(np.array(shapv), featv, show=False)  # = feature_mean
    save_as_svg(date_time, 'XGBoost_survival_summary_plot_full', subfolder=None)

    return

# -------------------------------------------------------------------
# Dependency with LOESS
# -------------------------------------------------------------------
def fig_dep_loess(shap_mean, feature_mean, topfeatures, hr_ci_df):
    """
    Generate and save SHAP dependence plots with LOESS smoothing for the top features,
    and update the corresponding hazard ratio (HR) and confidence interval (CI) dataframe.

    This function iterates through the top features and produces SHAP dependence plots,
    showing how each feature’s value affects its SHAP contribution while controlling for
    the effects of other top features. It leverages LOESS smoothing for continuous trends,
    enabling clear visualization of nonlinear effects.

    Parameters
    ----------
    shap_mean : pandas.DataFrame
        DataFrame containing mean SHAP values for each feature across samples.
        Each column corresponds to a feature, and each row to an observation.

    feature_mean : pandas.DataFrame
        DataFrame of the corresponding feature values for the same samples as in
        `shap_mean`. The columns must match those in `shap_mean`.

    topfeatures : list of str
        List of the most important feature names to include in the dependence analysis.

    hr_ci_df : pandas.DataFrame
        A dataframe containing the clinically chosen reference values for continuous variables.

    Returns
    -------
    hr_ci_df : pandas.DataFrame
        Updated hazard ratio and confidence interval dataframe, saved as an Excel file
        named 'HR_CI_cont'.

    Side Effects
    -------------
    - Calls `plot_shap_feature()` to generate SHAP dependence plots for each feature.
    - Saves the updated HR/CI dataframe to disk via `excel()` using the global `date_time`.
    - Produces and saves plots showing SHAP value vs. feature value relationships
      with LOESS smoothing.

    Notes
    -----
    - Each iteration produces a dependence plot for one feature while accounting
      for interactions with other top features.
    - Requires helper functions:
        * `plot_shap_feature()` — creates and returns updated HR/CI data after plotting.
        * `excel()` — saves the resulting dataframe to an Excel file.
    - The global variable `date_time` is used for file naming consistency.

    """
    if len(shap_mean.columns) > 1:
        for feature in topfeatures:
            feature_name = feature # Current feature
            all_features_but_feature_name = [x for x in topfeatures if x != feature] # All other features

            # Plot dependency plots - with LOESS for continuous variables and with a circle marking reference values for the plots in the article
            hr_ci_df = plot_shap_feature(shap_mean,
                                         feature_mean,
                                         feature_name,
                                         all_features_but_feature_name,
                                         hr_ci_df,
                                         n_plots=len(topfeatures) - 1)
    excel(date_time, hr_ci_df, 'HR_CI_cont')
    return

# -------------------------------------------------------------------
# Get top features
# -------------------------------------------------------------------
def top_features(shap_mean, nbr_of_features = 14):
    """
    Identify and rank the top features based on mean absolute SHAP values.

    This function calculates the mean absolute SHAP value for each feature, ranks
    all features by their contribution magnitude, logs the ranking to a file, and
    returns the top features in descending order of importance.

    Parameters
    ----------
    shap_mean : pandas.DataFrame
        DataFrame containing SHAP values for each feature across all samples.
        Each column represents a feature, each row an observation.

    nbr_of_features : int, optional (default=14)
        Number of top features to return based on mean absolute SHAP value.

    Returns
    -------
    topfeatures : list of str
        List of the top `nbr_of_features` feature names, sorted from highest to lowest
        mean absolute SHAP value.

    Side Effects
    -------------
    - Writes the ranking of all features to a file named 'all_variables_rank' using
      the global `date_time`.
    - Writes the top `nbr_of_features` features to a file named 'variable_rank'.
    - Prints the list of top features to the console.

    Notes
    -----
    - Ranking is based on **mean absolute SHAP value**, which reflects the overall
      contribution magnitude of each feature to the model predictions.
    - The function reverses the order after sorting to ensure the largest SHAP values
      are first.

    """
    # Sort features after SHAP size
    order = shap_mean.abs().mean().values.argsort()
    feature_names = shap_mean.columns.to_list()

    # Find the rank of ACEi and ARB
    descending_order = order.tolist()
    descending_order.reverse()
    for iii, idx in enumerate(descending_order):
        write_to_file(date_time, 'all_variables_rank', '\n' + str(feature_names[idx]) + ' is rank ' + str(iii) + ', mean abs SHAP value: ' + str(shap_mean[feature_names[idx]].abs().mean()))

    # Top 13
    tops = order[len(order) - nbr_of_features:].tolist()
    tops.reverse()  # get largest shap values first

    # Get all top 13 features
    topfeatures = []
    for iii, idx in enumerate(tops):
        topfeatures.append(feature_names[idx])
        write_to_file(date_time, 'variable_rank', '\n' + str(feature_names[idx]) + ' is rank ' + str(iii))

    print('Top ' + str(nbr_of_features) + ' features:', topfeatures)
    return topfeatures

# -------------------------------------------------------------------
# Kaplan-Meier and log rank test
# -------------------------------------------------------------------
def km(feature_mean, pred_and_targ):
    """
    Perform Kaplan-Meier survival analysis and log-rank test stratified by number of medications per patient.

    This function divides patients into two groups based on the number of medications:
    - Low medication: < 5 medications per patient
    - High medication: ≥ 5 medications per patient

    It then performs the following steps:
    1. Conducts a **log-rank test** to compare survival distributions between the two groups.
    2. Binarizes the 'meds_per_pat' feature to 0 (low) and 1 (high) for Kaplan-Meier plotting.
    3. Plots **Kaplan-Meier survival curves** for both groups.
    4. Saves the figure as an SVG using a timestamped file name.

    Parameters
    ----------
    feature_mean : pandas.DataFrame
        DataFrame containing at least the column 'meds_per_pat', which represents
        the number of medications per patient.

    pred_and_targ : pandas.DataFrame
        DataFrame containing survival data with columns:
        - 'survival_days_from_icu_start': Time to event or censoring.
        - 'mort': Event indicator (1 if the patient died, 0 if censored).

    Returns
    -------
    None
        The function prints the log-rank test results, plots the Kaplan-Meier curves,
        and saves the figure as an SVG file. No values are returned.

    Notes
    -----
    - The Kaplan-Meier curves are color-coded:
        - Blue: Low medication (< 5)
        - Red: High medication (≥ 5)
    - Uses `lifelines.KaplanMeierFitter` and `lifelines.statistics.logrank_test`.

    """
    ### Log rank test
    meds_high_idx = feature_mean.loc[feature_mean['meds_per_pat'] >= 5].index
    meds_low_idx = feature_mean.loc[feature_mean['meds_per_pat'] < 5].index
    meds_high = pred_and_targ.iloc[pred_and_targ.index.isin(meds_high_idx)]
    meds_low = pred_and_targ.iloc[pred_and_targ.index.isin(meds_low_idx)]
    meds_high['survival_days_from_icu_start'] = pred_and_targ['survival_days_from_icu_start'].iloc[pred_and_targ.index.isin(meds_high_idx)]
    meds_low['survival_days_from_icu_start'] = pred_and_targ['survival_days_from_icu_start'].iloc[pred_and_targ.index.isin(meds_low_idx)]
    log_rank_results = statistics.logrank_test(meds_low['survival_days_from_icu_start'],
                                               meds_high['survival_days_from_icu_start'], meds_low['mort'],
                                               meds_high['mort'])
    print(log_rank_results)

    ### Kaplan-Meier
    # Identify low and high medication patients and binarize into low (0) and high (1)
    feature_mean['meds_per_pat'].loc[
        feature_mean['meds_per_pat'] < 5] = 0
    feature_mean['meds_per_pat'].loc[
        feature_mean['meds_per_pat'] >= 5] = 1

    df = feature_mean.copy()
    df["time"] = pred_and_targ['survival_days_from_icu_start']
    df["event"] = pred_and_targ['mort']

    # Initialize plot
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    # Map label
    label_map = {
        0: "< 5",
        1: "≥ 5"
    }

    # Plot survival curve for each group
    colors = ['blue', 'red']
    for group in df["meds_per_pat"].unique():
        ix = df["meds_per_pat"] == group
        label = label_map.get(group, str(group))  #
        kmf.fit(df.loc[ix, "time"], event_observed=df.loc[ix, "event"],
                label=label)
        kmf.plot_survival_function(color = colors[group])

    plt.title("Kaplan-Meier curves by polypharmacy")
    plt.xlabel("Days")
    plt.ylabel("Survival (%)")
    plt.grid(True)
    plt.legend(title="Medications per patient")
    save_as_svg(date_time, 'Kaplan_Meier_meds_per_pat', subfolder=None)

    return

# -------------------------------------------------------------------
# Schoenfeld residuals of SHAP values
# -------------------------------------------------------------------
def schoenfeld_residuals(shap_mean, pred_and_targ, topfeatures: list):
    """
    Assess the proportional hazards assumption using Schoenfeld residuals for top SHAP features.
    This approach assumes SHAP values behave like linear covariates for proportional hazards testing.

    This function performs the following steps:
    1. Renames SHAP feature columns using a predefined mapping (`change_names`) for better readability.
    2. Subsets the SHAP values to include only the specified `topfeatures`.
    3. Adds survival information from `pred_and_targ`:
       - 'survival_days_from_icu_start': Time to event or censoring.
       - 'mort': Event indicator (1 if death occurred, 0 if censored).
    4. Fits a **Cox Proportional Hazards model** using SHAP values as covariates.
    5. Checks the **proportional hazards assumption** using Schoenfeld residuals.
       - Outputs plots and warnings for any features that violate the assumption (p-value < 0.05).

    Parameters
    ----------
    shap_mean : pandas.DataFrame
        DataFrame containing SHAP values for all features.

    pred_and_targ : pandas.DataFrame
        DataFrame containing survival data with columns:
        - 'survival_days_from_icu_start': Time to event or censoring.
        - 'mort': Event indicator (1 if death occurred, 0 if censored).

    topfeatures : list
        List of top features (column names in `shap_mean`) to include as covariates in the Cox model.

    Returns
    -------
    None
        The function fits a CoxPH model and checks assumptions in-place.
        It outputs plots and console messages indicating which variables (if any) fail the proportional hazards assumption.

    Notes
    -----
    - Uses `lifelines.CoxPHFitter` and its `check_assumptions` method for proportional hazards testing.
    - SHAP values are used as covariates, allowing interpretability of feature contributions in survival analysis.
    """
    # Rename variables for the plots
    shap_mean.rename(columns = change_names, inplace=True)
    _, topfeatures_renamed = plot_names(change_names, 'random_word', topfeatures)

    # Create a DataFrame of SHAP values to use as covariates (only for top12 or for all?)
    shap_prop_hazards = shap_mean[topfeatures_renamed]
    shap_prop_hazards['survival_days_from_icu_start'] = pred_and_targ['survival_days_from_icu_start']
    shap_prop_hazards['mort'] = pred_and_targ['mort']

    # Fit a CoxPH model using SHAP values as covariates
    cph = CoxPHFitter()
    cph.fit(shap_prop_hazards, duration_col='survival_days_from_icu_start', event_col='mort', show_progress=True)

    # Check proportional hazards assumption
    cph.check_assumptions(shap_prop_hazards, p_value_threshold=0.05, show_plots=True)
    # Variable 'meds_per_pat' failed the non-proportional test: p-value is 0.0003. It is not a surprise since...

    return


# -------------------------------------------------------------------
# C-index at different times
# -------------------------------------------------------------------
def prop_hazards(pred_and_targ):
    """
        Evaluate the time-dependent concordance index (C-index) for survival predictions.

        This function computes the C-index at multiple time points (3, 6, 9, 12 months)
        using both inverse probability of censoring weighting (IPCW) and bootstrap resampling
        to estimate 95% confidence intervals. The C-index measures the discriminatory power
        of predicted survival risks.

        Parameters
        ----------
        pred_and_targ : pandas.DataFrame
            DataFrame containing survival outcomes and predicted risk scores. Required columns:
            - 'mort': Event indicator (1 if death occurred, 0 if censored).
            - 'survival_days_from_icu_start': Time to event or censoring in days.
            - 'validation_predictions': Predicted risk scores from a model.

        Returns
        -------
        None

        Notes
        -----
        - Uses structured arrays compatible with `sksurv` for IPCW computation.
        - Bootstrap resampling is performed `RESAMPLE` times to estimate confidence intervals.
        - Time points are specified in days: 90, 180, 270, 365 (~3, 6, 9, 12 months).
        - The C-index evaluates how well the predicted risks rank patients by survival.
         """

    # Structured arrays
    y_train_structured = Surv.from_arrays(pred_and_targ["mort"].astype(bool).values, pred_and_targ["survival_days_from_icu_start"].values)
    y_test_structured = Surv.from_arrays(pred_and_targ["mort"].astype(bool).values, pred_and_targ["survival_days_from_icu_start"].values)

    # Evaluate C-index at 3, 6, 9, 12 months (in days)
    time_points = [90, 180, 270, 365]

    print("C-index at different time points:")
    for t in time_points:
        c_index, _, _, _, _ = concordance_index_ipcw(
            y_train_structured,
            y_test_structured,
            pred_and_targ['validation_predictions'],
            tau=t
        )

        # CI for C-index
        val_c = []
        for re in range(RESAMPLE):
            bs_sample = pred_and_targ.sample(n=len(pred_and_targ),
                                                       replace=True)  # Sampling with replacement
            cindex_val = concordance_index_censored(bs_sample['mort'].astype(bool),
                                                    bs_sample['survival_days_from_icu_start'],
                                                    bs_sample['validation_predictions'])
            val_c.append(cindex_val[0])
        ci_val = st.t.interval(0.95, len(val_c) - 1, loc=c_index, scale=st.sem(val_c))

        print(f"{t} days ({t//30} months): C-index = {c_index:.3f}" + ' (' + str(round(ci_val[0], 3)) + '-' + str(
        round(ci_val[1], 3)) + ')')
        write_to_file(date_time, 'c-index_times', f"{t} days ({t//30} months): C-index = {c_index:.3f}" + ' (' + str(round(ci_val[0], 3)) + '-' + str(
        round(ci_val[1], 3)) + ')/n', 'c-index')

    return



# -------------------------------------------------------------------
# Import data files
# -------------------------------------------------------------------
shap_val_all = pd.read_excel(r"all_internal_validation_SHAP_values",
                          engine='openpyxl')
shap_val_mean =pd.read_excel(r"mean_internal_validation_SHAP_values",
                          engine='openpyxl')
feature_val_all = pd.read_excel(r"all_feature_values_over_imputed_datasets",
                          engine='openpyxl')
feature_val_mean = pd.read_excel(r"mean_feature_values",
                          engine='openpyxl')
predictions_and_targets = pd.read_excel(r"risk_scores_and_true_survival_times",
                          engine='openpyxl')

# Drop extra columns
if 'Unnamed: 0' in shap_val_all.columns.to_list():
    shap_val_all = shap_val_all.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in shap_val_mean.columns.to_list():
    shap_val_mean = shap_val_mean.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in feature_val_all.columns.to_list():
    feature_val_all = feature_val_all.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in feature_val_mean.columns.to_list():
    feature_val_mean = feature_val_mean.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in predictions_and_targets.columns.to_list():
    predictions_and_targets = predictions_and_targets.drop('Unnamed: 0', axis=1)

# -------------------------------------------------------------------
# Run functions for figs and tests
# -------------------------------------------------------------------
top14features = top_features(shap_val_mean, 14)
fig_summary(shap_val_mean, feature_val_mean, 14)
points_of_interest_df = shap_hr(shap_val_all, feature_val_all, top14features, just_get_df = False)
schoenfeld_residuals(shap_val_mean, predictions_and_targets, top14features)
fig_dep_loess(shap_val_mean, feature_val_mean, top14features, points_of_interest_df)
km(feature_val_mean, predictions_and_targets)
prop_hazards(predictions_and_targets)

# -------------------------------------------------------------------
# Save all figs in a common PDF file
# -------------------------------------------------------------------
pdf(date_time, 'figs', None)