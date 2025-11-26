import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ==== 1. Set page configuration and title ====
st.set_page_config(page_title="Histogram Fitter Webapp", layout="centered")
st.title("Histogram Fitter Webapp")
st.markdown(
    """
    Upload a CSV file, paste numbers, fit a distribution, and adjust parameters for manual fitting.  
    Visualize your data and evaluate fit quality!
    """
)

# ==== 2. Layout: Tabs for cleaner interface ====
tab1, tab2 = st.tabs(["Data Entry", "Distribution Fitting"])

with tab1:
    st.header("Enter Your Data")
    data_source = st.radio("How would you like to input data?", ["Paste data", "Upload CSV"])
    
    data = None
    if data_source == "Paste data":
        raw_data = st.text_area(
            "Paste your data here (one value per line, or separated by comma/space)",
            height=150,
            placeholder="Example:\n1.3\n2.8\n4.2\n..."
        )
        if raw_data:
            # Try comma, space or newline separated values
            try:
                cleaned = raw_data.replace(',', ' ').replace('\n', ' ')
                vec = [float(x) for x in cleaned.split() if x.strip() != ""]
                data = np.array(vec)
            except Exception:
                st.error("Please enter valid numbers only.")
    else:
        uploaded_file = st.file_uploader("Choose a CSV (one column, or header optional)", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[1] == 1:
                    data = df.values.flatten()
                else:
                    col = st.selectbox("Pick which column:", df.columns)
                    data = df[col].values
            except Exception:
                st.error("Could not read the CSV file. Please check your format.")

    # Immediate summary feedback
    if data is not None and len(data) > 0:
        st.success(f"Loaded {len(data)} data points.")
        st.write("**Five number summary:**", 
                 pd.Series(data).describe()[['min', '25%', '50%', '75%', 'max']])
    elif data is not None:
        st.warning("No data detected.")

# ===== Main logic for fitting =====
with tab2:
    st.header("Distribution Fitting")
    if data is None or len(data) < 5:
        st.info("Please provide at least 5 valid data points in the 'Data Entry' tab.")
        st.stop()

    # List of distributions (at least 10)
    DISTROS = {
        'Normal (Gaussian)': stats.norm,
        'Gamma': stats.gamma,
        'Weibull': stats.weibull_min,
        'Beta': stats.beta,
        'Log-Normal': stats.lognorm,
        'Exponential': stats.expon,
        'Chi-Squared': stats.chi2,
        'Cauchy': stats.cauchy,
        'Laplace': stats.laplace,
        'Logistic': stats.logistic,
        'Rayleigh': stats.rayleigh,
        'Pareto': stats.pareto
    }
    distro_names = list(DISTROS.keys())

    selected = st.selectbox("Choose a distribution to fit:", distro_names)
    scipy_dist = DISTROS[selected]
    st.write(f"Selected Distribution: **{selected}**")

    # Fit parameters automatically
    try:
        fit_params = scipy_dist.fit(data)
        # Get parameter names (from scipy signature)
        param_names = scipy_dist.shapes.split(', ') if scipy_dist.shapes else []
        param_names += ['loc', 'scale']
        # Truncate param_names to length of fit_params (sometimes shapes=None)
        param_names = param_names[:len(fit_params)]
    except Exception as e:
        st.error(f"Could not fit distribution: {e}")
        st.stop()

    st.subheader("Fitted Parameters")
    fit_param_tbl = pd.DataFrame({
        "Parameter": param_names,
        "Estimate": fit_params
    })
    st.dataframe(fit_param_tbl, hide_index=True, use_container_width=True)

    # Manual override
    manual = st.checkbox("🔧 Manual Fitting (Adjust parameters by hand)", value=False)
    manual_params = list(fit_params)
    if manual:
        for i, (pname, pval) in enumerate(zip(param_names, fit_params)):
            # Try to have a reasonable slider range
            minv, maxv = (pval*0.5, pval*1.5) if abs(pval) > 0 else (-10, 10)
            if pname == "scale":
                minv = min(0.001, maxv)
            manual_params[i] = st.slider(
                f"{pname}", float(minv), float(maxv) if maxv > minv else float(minv)+1, float(pval)
            )
    
    # Prepare x for plotting
    bins = st.slider("Number of histogram bins", 10, 100, 25)
    x_hist = np.linspace(np.nanmin(data), np.nanmax(data), bins) if len(data)>0 else np.linspace(0,1,25)
    x_range = st.slider("PDF X-axis range", float(np.nanmin(data)*.95), float(np.nanmax(data)*1.05), 
                        (float(np.nanmin(data)*.95), float(np.nanmax(data)*1.05)))
    x_plot = np.linspace(x_range[0], x_range[1], 200)

    # Prepare fitted distribution for plotting
    try:
        dist_obj = scipy_dist(*manual_params)
        pdf_vals = dist_obj.pdf(x_plot)
    except Exception:
        pdf_vals = np.zeros_like(x_plot)
        st.warning("Could not evaluate PDF with selected parameters.")

    # ================== Plot Area ====================
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(data, bins=bins, density=True, alpha=0.4, color='skyblue', label='Data Histogram')
    ax.plot(x_plot, pdf_vals, 'r-', lw=2, label='Fitted PDF')
    ax.set_xlim(x_range)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)

    # ============= Fitting Error Calculation ============
    # For metric, use: Avg absolute error of histogram bin heights vs. PDF evaluated at bin centers
    hist_vals, bin_edges = np.histogram(data, bins=bins, range=x_range, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        fit_at_bins = scipy_dist(*manual_params).pdf(bin_centers)
        abs_err = np.abs(hist_vals - fit_at_bins)
        mae = np.mean(abs_err)
        max_err = np.max(abs_err)
    except Exception:
        mae = np.nan
        max_err = np.nan
    st.subheader("Fit Quality")
    st.metric("Mean Absolute Error (Histogram vs. Fit)", f"{mae:.5f}")
    st.metric("Maximum Error (Histogram vs. Fit)", f"{max_err:.5f}")

    with st.expander("Show bin-by-bin comparison details"):
        st.dataframe(pd.DataFrame({
            "Bin center": bin_centers,
            "Histogram density": hist_vals,
            "Fit density": fit_at_bins,
            "Abs Error": abs_err
        }).round(5), use_container_width=True)

st.caption("Built with Streamlit, numpy, scipy, matplotlib, and pandas ✨")
st.caption("© 2024 Histogram Fitting App | By [Nithipan Sivakanthan]")
