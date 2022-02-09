from tkinter import W
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from matplotlib import rc

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Times New Roman'], 'size': 14})
rc('text', usetex=True)

FIGSIZE = (6, 4)

props = dict(facecolor='white', alpha=1)


def plot_mef(ba, df_elec, df_co2, which='generation'):
    """
    Plot the MEF for a given BA

    Parameters
    ----------
    ba: tag for the given BA (e.g. CISO)
    df_elec: electricity dataset
    df_co2: CO2 dataset
    which: what regressor to use
    """

    # compute MEF
    preds, mef, r2, (ba_, ba_co2), xlabel = compute_mef(
        ba, df_elec, df_co2, which=which)

    # plot
    _, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    ax.scatter(ba_, ba_co2, s=1)
    ax.plot(ba_.values, preds, color='darkorange')
    ax.set_xlabel(r'$\Delta$'+f'{xlabel} [MWh/h]')
    ax.set_ylabel(r'$\Delta\ CO_2$ [kg/h]')
    # place a text box in upper left in axes coords
    x_text = ba_.quantile(.0)
    y_text = ba_co2.quantile(.99)
    textstr = rf'$R^2$: {np.around(r2, 2)}' + \
        '\n' + rf'MEF: {np.around(mef, 2)} kg/MWh'
    ax.text(x_text, y_text, textstr, bbox=props, fontsize=12)
    plt.grid()
    plt.savefig(os.path.join("figs", f"{ba}_{which}.pdf"))

    return (ba_, ba_co2), mef, r2


def compute_mef(ba, df_elec, df_co2, which='generation'):
    """
    Compute average mefs (i.e. one value across all timepoints)
    with a simple linear regression.

    Parameters
    ----------
    ba: tag for the Balancing Authority
    df_elec: dataset for electricity
    df_co2: dataset for co2
    which: tag specifying which regressor to use
    """

    (ba_, ba_co2), xlabel = extract_cols(ba, df_elec, df_co2, which=which)
    X, y = ba_.values.reshape(-1, 1), ba_co2.values
    reg = LinearRegression(fit_intercept=True).fit(X, y)

    return reg.predict(X), reg.coef_[0], reg.score(X, y), (ba_, ba_co2), xlabel


def compute_hourly_mef(ba, df_elec, df_co2, which='generation'):
    """
    Compute average mefs (i.e. one value across all timepoints)
    with a simple linear regression.

    Parameters
    ----------
    ba: tag for the Balancing Authority
    df_elec: dataset for electricity
    df_co2: dataset for co2
    which: tag specifying which regressor to use
    """

    (ba_, ba_co2), _ = extract_cols(ba, df_elec, df_co2, which=which)
    mefs_hr = []
    r2s_hr = []

    # Iterate over the different hours in the day
    for hr in range(24):
        idx_hr = ba_.index.hour == hr
        ba_crt = ba_[idx_hr]
        ba_co2_crt = ba_co2[idx_hr]
        X, y = ba_crt.values.reshape(-1, 1), ba_co2_crt.values
        reg = LinearRegression(fit_intercept=True).fit(X, y)
        mefs_hr.append(reg.coef_[0])
        r2s_hr.append(reg.score(X, y))

    return mefs_hr, r2s_hr


def get_mef_distribution(df_elec, df_co2, which='generation'):
    """
    Compute the MEF for every BA]

    Parameters
    ----------
    df_elec: electricity dataset
    df_co2: CO2 dataset
    which: what type of regressor to use
    """

    BAs = get_BAs(df_co2)
    # compute the MEFs for every BA
    which = "generation"

    mefs = []
    r2s = []
    for ba in BAs:
        _, mef, r2, _, _ = compute_mef(ba, df_elec, df_co2, which=which)
        mefs.append(mef)
        r2s.append(r2)

    return BAs, mefs, r2s


def extract_cols(ba, df_elec, df_co2, which='generation'):
    """
    Extract the relevant column from the dataset for a given BA. 

    Parameters
    ----------
    ba: tag for the Balancing Authority
    df_elec: dataset for electricity
    df_co2: dataset for co2
    which: tag specifying which regressor to use
    """

    D_elec_col = f'EBA.{ba}-ALL.D.H'
    D_co2_col = f'CO2_{ba}_D'
    NG_elec_col = f'EBA.{ba}-ALL.NG.H'
    NG_co2_col = f'CO2_{ba}_NG'
    WND_col = f'EBA.{ba}-ALL.NG.WND.H'
    SUN_col = f'EBA.{ba}-ALL.NG.SUN.H'

    if which == 'generation':
        ba_ = df_elec[NG_elec_col]
        ba_co2 = df_co2[NG_co2_col]
        xlabel = 'Generation'
    elif which == 'net_generation':
        ba_ = df_elec[NG_elec_col]
        ba_co2 = df_co2[NG_co2_col]
        if WND_col in df_elec.columns:
            ba_ = ba_-df_elec[WND_col]
        if SUN_col in df_elec.columns:
            ba_ = ba_-df_elec[SUN_col]
        xlabel = 'Net Generation'
    elif which == 'demand':
        ba_ = df_elec[D_elec_col]
        ba_co2 = df_co2[D_co2_col]
        xlabel = 'Demand'
    elif which == 'net_demand':
        ba_ = df_elec[D_elec_col]
        ba_co2 = df_co2[D_co2_col]
        if WND_col in df_elec.columns:
            ba_ = ba_-df_elec[WND_col]
        if SUN_col in df_elec.columns:
            ba_ = ba_-df_elec[SUN_col]
        xlabel = 'Net Demand'

    # extracting the appropriate columns
    idx = ba_.index.intersection(ba_co2.index)

    # compute hourly changes
    ba_ = ba_.loc[idx]
    ba_co2 = ba_co2.loc[idx]
    ba_ = ba_.diff()
    ba_co2 = ba_co2.diff()

    ba_ = ba_.loc[~ba_.isna()]
    ba_co2 = ba_co2.loc[~ba_co2.isna()]

    return (ba_, ba_co2), xlabel


def get_BAs(df):
    """
    Extract the list of all BAs present in the dataset. 

    Parameters
    ----------
    df: dataset
    """

    # extracting the names of the BAs present in the dataset
    nms = []
    fields = []

    for c in df.columns:

        nms.append(c.split('_')[1])
        fields.append(c.split('_')[-1])

    BAs = []
    for nm in nms:
        if '-' in nm:
            pass
        else:
            BAs.append(nm)

    return np.unique(BAs)
