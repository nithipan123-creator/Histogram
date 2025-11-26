#import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st



st.set_page_config(page_title="Histogram Fitting Webapp", layout="wide")


DISTRIBUTIONS = {
    "Normal (Gaussian)": stats.norm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Exponential": stats.expon,
    "Beta": stats.beta,
    "Log-Normal": stats.lognorm,
    "Uniform": stats.uniform,
    "Student's t": stats.t,
    "Cauchy": stats.cauchy,
    "Laplace": stats.laplace,
    "Logistic": stats.logistic,
}



def parse_manual_data(raw_text: str) -> np.ndarray:
    """
    Turn a string like '1, 2 3\n4' into a 1D numpy array of numbers.
    """
    # Replace newlines with commas so genfromtxt can handle both commas and newlines
    cleaned = raw_text.replace("\n", ",")
    data = np.genfromtxt(cleaned, delimiter=",")
    # Force 1D and drop NaNs
    data = np.atleast_1d(data)
    return data[~np.isnan(data)]


def get_param_info(dist, params):
    """
    Given a SciPy distribution and its fitted parameters, return a list of
    (name, value) pairs for shape parameters, loc, and scale.
    """
    shape_names = dist.shapes.split(", ") if dist.shapes else []
    param_info = []

    n_params = len(params)
    for i, value in enumerate(params):
        if i < len(shape_names):
            name = shape_names[i]  # shape parameters (e.g. "a", "b")
        elif i == n_params - 2:
            name = "loc"
        elif i == n_params - 1:
            name = "scale"
        else:
            name = f"param_{i + 1}"
        param_info.append((name, value))

    return param_info


def make_param_sliders(param_info, data):
    """
    Create sliders for each parameter and return the values as a list.
    param_info: list of (name, default_value)
    """
    slider_values = []

    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_range = max(data_max - data_min, 1e-6)  # avoid zero range

    st.markdown("**Adjust the parameters manually:**")

    for name, default in param_info:
        # Decide slider range based on parameter type
        if name == "loc":
            # Location: a bit beyond data range
            minv = data_min - 0.5 * data_range
            maxv = data_max + 0.5 * data_range
        elif name == "scale":
            # Scale must be > 0, use a positive range based on data spread
            minv = data_range / 50
            maxv = data_range * 2
        else:
            # Shape parameters: generic range based on data
            minv = 0.0
            maxv = max(2 * data_max, 10.0)

        # Make sure default is inside [minv, maxv]
        value = float(np.clip(default, minv, maxv))
        step = (maxv - minv) / 200 if maxv > minv else 0.1

        slider_values.append(
            st.slider(
                label=name,
                min_value=float(minv),
                max_value=float(maxv),
                value=float(value),
                step=float(step),
            )
        )

    return slider_values


def plot_hist_with_pdf(data, dist, params, n_bins, title_label):
    """
    Draw a normalized histogram of 'data' and overlay the PDF of 'dist' with 'params'.
    Returns (fig, bin_centers, hist_vals) for error calculations.
    """
    x = np.linspace(np.min(data), np.max(data), 400)

    try:
        y = dist.pdf(x, *params)
    except Exception:
        y = np.zeros_like(x)

    # Histogram
    hist_vals, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=n_bins, density=True, alpha=0.6, label="Data Histogram")
    ax.plot(x, y, lw=2, label=title_label)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (normalized)")
    ax.legend()

    return fig, bin_centers, hist_vals


def compute_fit_error(dist, params, bin_centers, hist_vals):
    """
    Compute average and maximum absolute error between histogram and PDF.
    """
    try:
        curve_vals = dist.pdf(bin_centers, *params)
        abs_diff = np.abs(hist_vals - curve_vals)
        return float(np.mean(abs_diff)), float(np.max(abs_diff))
    except Exception:
        return None, None



st.sidebar.header("1. Data Input")

data_input_method = st.sidebar.radio(
    "Select data input method:",
    ("Enter by hand", "Upload CSV"),
)

data = None

if data_input_method == "Enter by hand":
    manual_text = st.sidebar.text_area(
        "Enter numbers separated by commas, spaces, or newlines:",
        value="10, 12, 11, 14, 13, 13, 12, 15, 14, 16",
    )
    try:
        data = parse_manual_data(manual_text)
    except Exception:
        st.sidebar.warning(
            "Please enter numbers separated by commas, spaces, or lines.\n"
            "Example: 1, 2, 3, 4"
        )
        data = np.array([])

else:  # Upload CSV
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV with a single column of numbers", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] > 1:
                st.sidebar.warning("Multiple columns detected. Using the first column only.")
            data = df.iloc[:, 0].values.astype(float)
        except Exception as e:
            st.sidebar.error(f"Error reading the uploaded file: {e}")
            data = np.array([])
    else:
        data = np.array([])

# If not enough data, stop here
if data is None or len(data) < 5:
    st.info("Please enter or upload at least 5 data points to begin.")
    st.stop()

st.sidebar.markdown("---")


st.title("Histogram Fitting Webapp")
tab_auto, tab_manual = st.tabs(["Fit Distributions (Auto)", "Manual Fitting"])


with tab_auto:
    st.header("2. Automatic Distribution Fitting")

    dist_name_auto = st.selectbox(
        "Select distribution to fit",
        list(DISTRIBUTIONS.keys()),
        index=0,
        key="auto_dist",
    )
    dist_auto = DISTRIBUTIONS[dist_name_auto]

    n_bins = st.slider("Number of histogram bins", 5, 75, 25)

    # Fit the distribution parameters to the data
    params_auto = dist_auto.fit(data)
    param_info_auto = get_param_info(dist_auto, params_auto)

    # Plot histogram + PDF
    fig_auto, bin_centers_auto, hist_vals_auto = plot_hist_with_pdf(
        data,
        dist_auto,
        params_auto,
        n_bins,
        title_label=f"{dist_name_auto} (auto fit)",
    )
    st.pyplot(fig_auto)

    # Show parameters
    st.subheader("Fitted Parameters")
    pretty_params = ", ".join(
        f"{name} = {value:.4g}" for name, value in param_info_auto
    )
    st.code(pretty_params)

    # Show simple error metrics
    avg_err, max_err = compute_fit_error(
        dist_auto, params_auto, bin_centers_auto, hist_vals_auto
    )
    if avg_err is not None:
        st.info(f"Average absolute error (histogram vs. fit): {avg_err:.3g}")
        st.info(f"Maximum absolute error (histogram vs. fit): {max_err:.3g}")



with tab_manual:
    st.header("3. Manual Distribution Parameter Adjustment")

    dist_name_manual = st.selectbox(
        "Select distribution for manual fitting",
        list(DISTRIBUTIONS.keys()),
        key="manual_dist",
    )
    dist_manual = DISTRIBUTIONS[dist_name_manual]

    # Start sliders from the automatic fit as "good guesses"
    params_manual_default = dist_manual.fit(data)
    param_info_manual = get_param_info(dist_manual, params_manual_default)

    # Parameter sliders
    slider_params = make_param_sliders(param_info_manual, data)

    # Plot histogram + manual PDF
    fig_manual, bin_centers_manual, hist_vals_manual = plot_hist_with_pdf(
        data,
        dist_manual,
        slider_params,
        n_bins,
        title_label=f"{dist_name_manual} (manual fit)",
    )
    st.pyplot(fig_manual)

    # Error for manual parameters
    avg_err_m, max_err_m = compute_fit_error(
        dist_manual, slider_params, bin_centers_manual, hist_vals_manual
    )
    if avg_err_m is not None:
        st.info(f"Manual fit – average absolute error: {avg_err_m:.3g}")
        st.info(f"Manual fit – maximum absolute error: {max_err_m:.3g}")
    else:
        st.warning("Unable to compute fit error for these parameters.")






