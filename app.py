import streamlit as st
import pickle
import numpy as np
import pandas as pd
import traceback
import sys

# Try to load pickles safely and show a helpful Streamlit message instead of crashing the app
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)

    # --- begin added: normalize column names and map common variants ---
    def _norm(s):
        return ''.join(ch for ch in str(s).lower() if ch.isalnum())

    expected = {
        'Company': ['company'],
        'TypeName': ['typename', 'type'],
        'Ram': ['ram', 'memory'],
    'Weight': ['weight'],
        # expand variants that may represent an IPS/panel column
        'ips': ['ips', 'display', 'displaytype', 'screen_type', 'panel', 'screen', 'panel_type'],
        'ppi': ['ppi'],
        'Cpu brand': ['cpubrand', 'cpu', 'processor'],
        'hdd': ['hdd'],
        'ssd': ['ssd'],
        'Gpu brand': ['gpubrand', 'gpu'],
        'os': ['os', 'operatingsystem', 'operating_system']
    }

    col_map = {}
    cols_norm = { _norm(c): c for c in df.columns }

    for canonical, variants in expected.items():
        found = None
        for v in variants:
            nv = _norm(v)
            if nv in cols_norm:
                found = cols_norm[nv]; break
        if found is None:
            # try fuzzy substring match as a fallback
            for n, orig in cols_norm.items():
                for v in variants:
                    if _norm(v) in n:
                        found = orig; break
                if found:
                    break
        if found:
            col_map[found] = canonical

    if col_map:
        df = df.rename(columns=col_map)
    # verify required columns exist, otherwise show helpful error
    required = ['Company','TypeName','Ram','Weight','ips','ppi','Cpu brand','hdd','ssd','Gpu brand','os']
    missing = [c for c in required if c not in df.columns]

    # If ips is missing, create it with a safe default (non-IPS) and warn the user.
    if 'ips' in missing:
        df['ips'] = 0
        st.warning(
            "Column 'ips' not found in df.pkl — defaulting 'ips' to 0 (assumes non-IPS). "
            "If your saved dataframe contains IPS information under a different name, rename that column before saving df.pkl."
        )

    # (Touchscreen removed) do not create or expect Touchscreen in df

    # recompute missing after adding defaults
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.title("Laptop Predictor")
        st.error("Required columns missing from df.pkl: " + ", ".join(missing))
        st.markdown("Available columns: " + ", ".join(list(df.columns)))
        st.markdown("If your df has different names, rename them before saving or adjust the app's expected names.")
        st.stop()
    # --- end added ---
except Exception as e:
    st.title("Laptop Predictor")
    st.error("Failed to load model or dataframe (pipe.pkl / df.pkl).")
    st.markdown(
        "Likely causes: mismatch between scikit-learn / numpy versions used to create the pickle "
        "and the ones in this environment, corrupted files, or Python version incompatibility."
    )
    st.markdown("Quick fixes:")
    st.markdown("- Recreate the virtualenv with the same Python + package versions used to save the model.")
    st.markdown("- Retrain / re-save the model in the current environment (prefer joblib.dump).")
    st.markdown("- If you must unpickle, install the exact versions used to create the files.")
    # targeted guidance for missing modules (common during unpickle)
    if isinstance(e, ModuleNotFoundError):
        missing = getattr(e, "name", None) or str(e)
        st.markdown(f"Detected missing module: `{missing}`.")
        install_cmd = f"{sys.executable} -m pip install {missing} --prefer-binary"
        st.markdown("Install the missing package into your project's environment, for example:")
        st.code(install_cmd)
        st.markdown("If installation fails on macOS (Apple Silicon), try: `pip install xgboost --prefer-binary` or use conda from conda-forge.")
    st.markdown("Full error:")
    st.code(traceback.format_exc())
    # stop the Streamlit script here (avoids terminal tracebacks)
    st.stop()

# Try to infer the pipeline's expected input column names and provide
# utilities to map our app's query DataFrame to those expected names.
def _infer_pipe_input_columns(pipe):
    """Return a list of column names that the pipeline expects as input.

    Tries several strategies (best-effort):
    - attribute `feature_names_in_` on pipeline or transformer
    - find a ColumnTransformer inside a Pipeline and read its transformers_
    - fall back to empty list if nothing detected
    """
    try:
        # direct attribute used by many sklearn estimators
        cols = []
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)

        # avoid importing sklearn at module import time if not available;
        # import locally and be tolerant to import errors
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
        except Exception:
            Pipeline = None
            ColumnTransformer = None

        ct = None
        # if it's a Pipeline, scan its steps for a ColumnTransformer
        if Pipeline is not None and isinstance(pipe, Pipeline):
            for _name, step in pipe.steps:
                if ColumnTransformer is not None and isinstance(step, ColumnTransformer):
                    ct = step
                    break
                # some pipelines wrap transformers inside objects that expose transformers_
                if hasattr(step, "transformers_"):
                    ct = step
                    break

        # if not a Pipeline, maybe the pipe itself is a ColumnTransformer-like
        if ct is None and hasattr(pipe, "transformers_"):
            ct = pipe

        if ct is not None:
            for name, transformer, cols_spec in ct.transformers_:
                # skip explicit drops
                if cols_spec == 'drop' or name == 'drop':
                    continue
                # cols_spec can be list-like, ndarray, slice, string, or callable
                try:
                    # list/tuple/ndarray
                    if isinstance(cols_spec, (list, tuple)):
                        cols = cols + list(cols_spec)
                    elif hasattr(cols_spec, 'tolist'):
                        cols = cols + list(cols_spec.tolist())
                    elif isinstance(cols_spec, str):
                        cols.append(cols_spec)
                except Exception:
                    # ignore anything we can't convert
                    continue

        # remove duplicates while preserving order
        if cols:
            seen = set()
            out = []
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
    except Exception:
        pass
    return []


def _map_query_to_pipeline(query_df, expected_pipe_cols):
    """Map the app's canonical column names to the pipeline's expected names.

    - query_df: pandas DataFrame built by the app (with canonical names)
    - expected_pipe_cols: list of column names expected by the pipeline

    Returns: (renamed_df, mapping_dict)
    """
    def _norm(s):
        return ''.join(ch for ch in str(s).lower() if ch.isalnum())

    # canonical column names used by this app
    canonical_cols = ['Company','TypeName','Ram','Weight','ips','ppi','Cpu brand','hdd','ssd','Gpu brand','os']
    canon_norm = { _norm(c): c for c in canonical_cols }

    mapping = {}
    for exp in expected_pipe_cols:
        en = _norm(exp)
        # direct canonical match
        if en in canon_norm:
            mapping[canon_norm[en]] = exp
            continue
        # heuristics for common variants
        if 'cpu' in en or 'processor' in en:
            mapping['Cpu brand'] = exp
        elif 'gpu' in en or 'graphics' in en:
            mapping['Gpu brand'] = exp
        elif 'ssd' in en or 'solid' in en:
            mapping['ssd'] = exp
        elif 'hdd' in en or 'hard' in en:
            mapping['hdd'] = exp
        elif 'mem' in en or 'ram' in en or 'memory' in en:
            mapping['Ram'] = exp
        # touchscreen removed from app canonical names
        elif 'ips' in en or 'panel' in en or 'display' in en:
            mapping['ips'] = exp
        elif 'ppi' in en or 'res' in en or 'resolution' in en:
            mapping['ppi'] = exp
        elif 'company' in en or 'brand' in en:
            mapping['Company'] = exp
        elif 'type' in en:
            mapping['TypeName'] = exp
        elif 'os' in en or 'operatingsystem' in en:
            mapping['os'] = exp
        elif 'weight' in en:
            mapping['Weight'] = exp

    # perform the rename (only for mappings we found)
    rename_map = {src: dst for src, dst in mapping.items() if src in query_df.columns}
    renamed = query_df.rename(columns=rename_map)

    # ensure all expected_pipe_cols are present: add defaults when missing
    for col in expected_pipe_cols:
        if col not in renamed.columns:
            # default numeric -> 0, default categorical -> take first available value if possible
            renamed[col] = 0
    # order columns to match expected_pipe_cols (predictors may require order for some custom transformers)
    try:
        renamed = renamed[expected_pipe_cols]
    except Exception:
        # if ordering fails, just return renamed as-is
        pass

    return renamed, rename_map


st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop (avoid shadowing builtin 'type')
type_name = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# weight (provide a sensible default and min)
weight = st.number_input('Weight of the Laptop (kg)', min_value=0.1, value=1.5, step=0.1)

# Touchscreen
#touchscreen = st.selectbox('Touchscreen', ['No','Yes'])

# IPS
ips = st.selectbox('IPS', ['No','Yes'])

# screen size
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

# cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# avoid shadowing builtin 'os'
os_name = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # convert categorical widgets to numeric flags
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # build a DataFrame matching training feature names and order
    query_data = pd.DataFrame(
        [[company, type_name, ram, weight, ips_val, ppi, cpu, hdd, ssd, gpu, os_name]],
        columns=['Company', 'TypeName', 'Ram', 'Weight', 'ips', 'ppi', 'Cpu brand', 'hdd', 'ssd', 'Gpu brand', 'os']
    )
    # try to remap the query columns to match what the saved pipeline expects
    try:
        expected_cols = _infer_pipe_input_columns(pipe)
    except Exception:
        expected_cols = []

    if expected_cols:
        try:
            query_for_pipe, rename_map = _map_query_to_pipeline(query_data, expected_cols)
            # fill any still-missing expected columns with safe defaults (prefer values from df if available)
            for col in expected_cols:
                if col not in query_for_pipe.columns:
                    if col in df.columns and not df[col].empty:
                        try:
                            query_for_pipe[col] = df[col].mode().iloc[0]
                        except Exception:
                            query_for_pipe[col] = 0
                    else:
                        query_for_pipe[col] = 0
            # ensure order
            try:
                query_for_pipe = query_for_pipe[expected_cols]
            except Exception:
                pass
        except Exception:
            # fallback to the original query_data if mapping fails
            query_for_pipe = query_data
    else:
        query_for_pipe = query_data

    try:
        pred_log = pipe.predict(query_for_pipe)[0]
        prediction = np.exp(pred_log)
        st.success(f"The predicted price of this configuration is {int(prediction)}")
    except Exception:
        st.error("Prediction failed. This is often caused by a pipeline/model pickle that is incompatible with installed package versions or mismatched column names.")
        st.code(traceback.format_exc())