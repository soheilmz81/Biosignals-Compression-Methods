import math
import numpy as np
import scipy.io as sio
import streamlit as st
import plotly.graph_objects as go

# ===========================================================================
#  SECTION 1 - Numeric Core (الگوریتم‌های فشرده‌سازی - بدون تغییر)
# ===========================================================================

def level_trigger(x, r0):
    kept = [0]
    anchor = float(x[0])
    for i in range(1, len(x)):
        if abs(x[i] - anchor) > r0:
            kept.append(i)
            anchor = float(x[i])
    if kept[-1] != len(x) - 1:
        kept.append(len(x) - 1)
    return np.asarray(kept, dtype=int)

def slope_projection(x, fs, r1):
    n = len(x)
    if n < 3: return np.arange(n)
    sdot = np.zeros(n)
    sdot[1:] = np.diff(x) * fs
    sel = np.abs(np.diff(sdot)) > r1
    kept = np.concatenate(([0], np.flatnonzero(sel) + 1, [n - 1]))
    return np.unique(kept)

def second_diff_gate(x, fs, r_l):
    n = len(x)
    if n < 3: return np.arange(n)
    s = np.diff(x) * fs
    sel = np.abs(s[1:] - s[:-1]) > r_l
    kept = np.concatenate(([0], np.flatnonzero(sel) + 1, [n - 1]))
    return np.unique(kept)

def turning_points(x):
    n = len(x)
    kept = [0]
    i = 0
    while i + 2 < n:
        s1 = np.sign(x[i + 1] - x[i])
        s2 = np.sign(x[i + 2] - x[i + 1])
        if s1 != 0 and s1 + s2 == 0:
            kept.append(i + 1)
        else:
            kept.append(i + 2)
        i += 2
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return np.asarray(kept, dtype=int)

def fan_envelope(x, eps):
    n = len(x)
    kept = [0]
    a = 0
    while a < n - 1:
        c = a + 1
        up = (x[c] + eps - x[a]) / (c - a)
        lo = (x[c] - eps - x[a]) / (c - a)
        e = c + 1
        while e < n:
            if x[e] > x[a] + up * (e - a) or x[e] < x[a] + lo * (e - a):
                break
            up = min(up, (x[e] + eps - x[a]) / (e - a))
            lo = max(lo, (x[e] - eps - x[a]) / (e - a))
            c = e
            e += 1
        kept.append(c)
        a = c
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return np.asarray(kept, dtype=int)

def segment_fit(x, t, m0):
    n = len(x)
    kept = [0]
    j = 0
    while j < n - 1:
        m = min(m0, n - 1 - j)
        while True:
            dx = x[j + m] - x[j]
            ks = np.arange(j + 1, j + m)
            if ks.size == 0:
                dmax, kworst = -np.inf, j
            else:
                d = (np.abs(m * (x[ks] - x[j]) - (ks - j) * dx)
                     / np.sqrt(m * m + dx * dx))
                i_max = int(np.argmax(d))
                dmax, kworst = d[i_max], int(ks[i_max])
            if dmax < t:
                kept.append(j + m)
                j += m
                break
            m = (kworst - j) - 1
            if m < 1:
                m = 1
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return np.asarray(kept, dtype=int)

def rebuild(x, kept, scheme):
    n = len(x)
    xs = x[kept]
    if scheme == 0:
        pos = np.searchsorted(kept, np.arange(n), side='right') - 1
        pos = np.clip(pos, 0, len(kept) - 1)
        rec = xs[pos].astype(float, copy=True)
    else:
        rec = np.interp(np.arange(n), kept, xs)
    rec[0], rec[-1] = x[0], x[-1]
    holes = np.flatnonzero(np.isnan(rec))
    for q in holes:
        k = np.searchsorted(kept, q, side='left') - 1
        rec[q] = x[kept[max(k, 0)]]
    return rec

def quality_numbers(x, xrec, n_kept):
    n = len(x)
    cr = n / n_kept
    e_rms = math.sqrt(np.nansum((x - xrec) ** 2) / n)
    v_rms = math.sqrt(np.nansum(x ** 2) / n)
    prd = (e_rms / v_rms * 100.0) if v_rms != 0 else float('nan')
    return cr, prd, e_rms, v_rms

# ===========================================================================
#  SECTION 2 - File Handling for Web (آپلود فایل در حافظه)
# ===========================================================================
def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.mat'):
        blob = sio.loadmat(uploaded_file)
        for key, val in blob.items():
            if not key.startswith('__'):
                v = np.asarray(val)
                if np.issubdtype(v.dtype, np.number):
                    v = np.squeeze(v)
                    if v.ndim == 1 and v.size > 0:
                        return v.astype(float)
    elif name.endswith('.csv') or name.endswith('.txt'):
        data = np.genfromtxt(uploaded_file, delimiter=',')
        if data.ndim == 2:
            return data[:, 1].astype(float) if data.shape[1] >= 2 else data[:, 0].astype(float)
        return data.astype(float)
    return None

# ===========================================================================
#  SECTION 3 - Streamlit Web UI (رابط کاربری تحت وب)
# ===========================================================================
st.set_page_config(page_title="PhysioCompact Studio", layout="wide")

st.title("PhysioCompact Studio (Web Version)")
st.markdown("A web workbench for reducing biosignals (ECG / EEG / EMG) with non-uniform sampling schemes.")

# --- Sidebar Controls ---
st.sidebar.header("1. Signal Source")
uploaded_file = st.sidebar.file_uploader("Upload Recording (MAT, CSV, TXT)", type=["mat", "csv", "txt"])
fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=250.0, min_value=1.0)

st.sidebar.header("2. Method and Threshold")
SCHEME_MENU = [
    'Order 0: Amplitude trigger',
    'Order 1: Slope projection',
    'Order 2: Second-difference gate',
    'TP: Turning-point decimation',
    'FAN: Fan envelope',
    'LADT: Segmented linear fit'
]
scheme_str = st.sidebar.selectbox("Reduction Scheme", SCHEME_MENU)
scheme_idx = SCHEME_MENU.index(scheme_str)

# Threshold inputs based on selected scheme
pv = [0.0]
if scheme_idx == 0:
    pv = [st.sidebar.number_input("Level threshold R0", value=0.05, format="%.3f")]
elif scheme_idx == 1:
    pv = [st.sidebar.number_input("Slope threshold R1", value=0.5, format="%.3f")]
elif scheme_idx == 2:
    pv = [st.sidebar.number_input("Curvature threshold R_L", value=0.5, format="%.3f")]
elif scheme_idx == 3:
    st.sidebar.info("Turning-point runs threshold-free.")
elif scheme_idx == 4:
    pv = [st.sidebar.number_input("Corridor half-width epsilon", value=0.05, format="%.3f")]
elif scheme_idx == 5:
    col1, col2 = st.sidebar.columns(2)
    t_val = col1.number_input("Budget T", value=0.5, format="%.3f")
    m0_val = col2.number_input("Length m0", value=20, min_value=1)
    pv = [t_val, m0_val]

btn_run = st.sidebar.button("Compress Signal")

# --- Main App Logic ---
if uploaded_file is not None:
    x = read_uploaded_file(uploaded_file)
    
    if x is not None:
        st.write(f"**Loaded signal:** {len(x)} samples.")
        
        if btn_run:
            t = np.arange(len(x)) / fs
            
            # 1. Run Compression
            if scheme_idx == 0: kept = level_trigger(x, pv[0])
            elif scheme_idx == 1: kept = slope_projection(x, fs, pv[0])
            elif scheme_idx == 2: kept = second_diff_gate(x, fs, pv[0])
            elif scheme_idx == 3: kept = turning_points(x)
            elif scheme_idx == 4: kept = fan_envelope(x, pv[0])
            elif scheme_idx == 5: kept = segment_fit(x, pv[0], pv[1])
            
            # 2. Rebuild and Score
            xrec = rebuild(x, kept, scheme_idx)
            cr, prd, e_rms, v_rms = quality_numbers(x, xrec, len(kept))
            
            # 3. Display Metrics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Compression Ratio (CR)", f"{cr:.3f} : 1")
            col2.metric("PRD (%)", f"{prd:.3f} %")
            col3.metric("Retained Samples", f"{len(kept)} / {len(x)}")
            col4.metric("RMSE", f"{e_rms:.4f}")
            
            # 4. Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Raw Signal', line=dict(color='#62b6cb', width=1)))
            fig.add_trace(go.Scatter(x=t, y=xrec, mode='lines', name='Rebuilt Trace', line=dict(color='#f28e2b', width=1.5)))
            fig.add_trace(go.Scatter(x=t[kept], y=x[kept], mode='markers', name='Retained Samples', marker=dict(color='#9be564', size=5, line=dict(color='black', width=1))))
            
            fig.update_layout(
                title=f"Compression Analysis - {SCHEME_MENU[scheme_idx]}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_dark",
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not read numeric data from the uploaded file.")
else:
    st.info("Please upload a signal file from the sidebar to begin.")