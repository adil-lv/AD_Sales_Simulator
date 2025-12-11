import pandas as pd
import streamlit as st
import requests
from streamlit_option_menu import option_menu
import plotly.express as px
import numpy as np
import pickle
import base64
import io
import os
from datetime import datetime
import json
from typing import Optional
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from typing import List, Dict
from dateutil import parser
import difflib
import re
import warnings
warnings.filterwarnings('ignore')
import time
from databricks.sdk import WorkspaceClient
from databricks import sql


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass
class TrainedSegment:
    segment_id: str
    best_name: str
    best_params: dict
    pipeline: Pipeline
    features_other: List[str]
    r2_train: float
    r2_val: float
    r2_test: float
    n_train: int
    n_val: int
    n_test: int


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)


# =============================================================================
# UNIVERSAL QUERY PARAMS HANDLING
# =============================================================================
def get_query_param(param_name, default=None):
    """Get query parameters with full backward compatibility"""
    try:
        if hasattr(st, 'query_params'):
            value = st.query_params.get(param_name, default)
            if value is not None:
                return value
    except:
        pass
    try:
        if hasattr(st, 'experimental_get_query_params'):
            query_params = st.experimental_get_query_params()
            if param_name in query_params:
                return query_params[param_name][0]
    except:
        pass
    return default


# =============================================================================
# CSP INJECTION
# =============================================================================
def add_csp():
    csp_script = """
    <script>
    const existingMeta = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
    if (existingMeta) {
        existingMeta.remove();
    }
    const meta = document.createElement('meta');
    meta.httpEquiv = 'Content-Security-Policy';
    meta.content = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:; " +
                   "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com https://cdn.plot.ly; " +
                   "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; " +
                   "font-src 'self' https://fonts.gstatic.com; " +
                   "connect-src 'self' https://adb-758907235096429.9.azuredatabricks.net https://*.azuredatabricks.net; " +
                   "frame-ancestors 'self' https://*.powerapps.com https://*.microsoft.com https://apps.powerapps.com; " +
                   "img-src 'self' data: https:;";
    document.getElementsByTagName('head')[0].appendChild(meta);
    </script>
    """
    st.components.v1.html(csp_script, height=0)


add_csp()

# =============================================================================
# LOGO AND CSS
# =============================================================================
def get_base64_of_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""


try:
    logo_base64 = get_base64_of_image("AD_Logo.png")
except:
    logo_base64 = ""


st.markdown(f"""
<style>
.stApp {{ background-color: #f0f2f6; }}
.main-header {{ background-color: white; padding: 10px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; align-items: center; justify-content: space-between; }}
.header-content {{ display: flex; align-items: center; width: 100%; justify-content: space-between; }}
.logo-container {{ display: flex; align-items: center; }}
.logo-img {{ height: 80px; width: 100px; margin-right: 20px; }}
.title-container {{ flex-grow: 1; text-align: center; }}
.header-title {{ color: #FF4500; font-size: 2.2em; font-weight: bold; margin: 0; }}
.main .block-container {{ max-width: 100% !important; padding-top: 1rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }}
[data-testid="column"] {{ width: 100% !important; }}
.metric-card {{ background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid #FF4500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 110px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 10px; }}
.metric-title {{ color: #FF4500; font-size: 14px; font-weight: 600; margin: 0 0 8px 0; }}
.metric-value {{ color: #333; font-size: 20px; font-weight: bold; margin: 0 0 5px 0; line-height: 1.2; }}
.metric-subtitle {{ color: #666; font-size: 11px; margin: 0; }}
div[data-baseweb="select"] > div {{ border: 1px solid #FF4500 !important; border-radius: 6px !important; }}
div[data-baseweb="input"] > div {{ border: 1px solid #FF4500 !important; border-radius: 6px !important; }}
.single-filter {{ max-width: 300px !important; }}
.stSlider [data-baseweb="slider"] {{ color: #FF4500 !important; }}
.stSlider [data-baseweb="slider"] [role="slider"] {{ background-color: #FF4500 !important; }}
.stButton button {{ background-color: #FF4500 !important; color: white !important; border: none !important; border-radius: 6px !important; }}
.stButton button:hover {{ background-color: #E03E00 !important; color: white !important; }}
.stRadio [role="radiogroup"] {{ border: 1px solid #FF4500 !important; border-radius: 6px !important; padding: 5px; }}
[data-testid="stSidebarNav"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
html, body, [class*="css"] {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; background-color: #f0f2f6; }}
.stTabs [data-baseweb="tab"] {{ background-color: white !important; border-radius: 4px 4px 0px 0px; padding: 12px 24px !important; border: none !important; color: #FF4500 !important; font-size: 16px !important; font-weight: 600 !important; margin: 0 2px !important; transition: all 0.3s ease !important; height: 50px !important; min-width: 180px !important; display: flex !important; align-items: center !important; justify-content: center !important; }}
.stTabs [data-baseweb="tab"]:hover {{ background-color: #FFE4D6 !important; color: #FF4500 !important; }}
.stTabs [aria-selected="true"] {{ background-color: #FF4500 !important; color: white !important; border: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ background-color: #f0f2f6; border-radius: 0px 0px 8px 8px; padding: 0px; margin-top: 0px; }}
.success-message {{ font-size: 14px; color: #00AA00; font-weight: 500; margin: 5px 0; }}
.model-info-message {{ font-size: 13px; color: #666; font-style: italic; margin: 2px 0 10px 0; }}
.feature-importance-title {{ color: #28a745; font-size: 16px; font-weight: 600; margin: 0 0 10px 0; text-align: center; }}
.feature-item {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 5px; border-bottom: 1px solid #f0f0f0; margin-bottom: 5px; }}
.feature-name {{ font-size: 13px; color: #333; flex: 1; margin-right: 10px; }}
.feature-importance {{ font-size: 12px; color: #FF4500; font-weight: 600; min-width: 60px; text-align: right; }}
.feature-adjustment-section {{ background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid #FF4500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; }}
.debug-panel {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 10px 0; max-height: 500px; overflow-y: auto; font-family: monospace; font-size: 12px; }}
.debug-step {{ margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #dee2e6; }}
.debug-step-title {{ font-weight: bold; margin-bottom: 5px; color: #333; }}
.debug-content {{ margin-left: 10px; }}
.debug-line {{ margin: 2px 0; }}
.debug-success {{ color: #28a745; font-weight: bold; }}
.debug-warning {{ color: #ffc107; font-weight: bold; }}
.debug-error {{ color: #dc3545; font-weight: bold; }}
.debug-info {{ color: #17a2b8; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATABRICKS CONFIGURATION
# =============================================================================
DATABRICKS_WORKSPACE_URL = "https://adb-758907235096429.9.azuredatabricks.net"
DATABRICKS_TOKEN = "dapic2a6dd56a02c5001c86be9916b27219d"

# from azure.identity import DefaultAzureCredential
# from azure.keyvault.secrets import SecretClient

# def get_databricks_token():
#     key_vault_url = "https://ai-sales-forecasting.vault.azure.net/"

#     credential = DefaultAzureCredential()
#     client = SecretClient(vault_url=key_vault_url, credential=credential)

#     secret = client.get_secret("databricks-token")
#     return secret.value


# DATABRICKS_TOKEN = get_databricks_token()


# =============================================================================
# FIXED FILE PATH CONFIGURATION WITH MONTH COLUMN INFO
# =============================================================================
FILE_CONFIG = {
    "country": {
        "China": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - China_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/RF_country_china.pkl",
            "features": [
                "asp",
                "cpi_commodities_less_food__energy_us",
                "cpi_apparel_us",
                "debt_credit_card_us",
                "cpi_services_less_energy_services_us"
            ],
            "month_column": "month"
        },
        "Bangladesh": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Bangladesh_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/XGB_country_bangladesh.pkl",
            "features": [
                "pet_film_price_index",
                "wage_growth_eu",
                "debt_auto_loan_us",
                "total_weight_supplied_us_bangladesh",
                "housing_starts_singlefamily_units_us"
            ],
            "month_column": "date"
        },
        "Honduras": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Honduras_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/XGB_country_honduras.pkl",
            "features": [
                "wage_growth_us",
                "gdp_real_eu",
                "pmi_honduras",
                "rfid_ic_price",
                "unemployment_eu"
            ],
            "month_column": "date"
        },
        "Turkey": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Turkey_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/GB_country_turkey.pkl",
            "features": [
                "cpi_services_less_energy_services_us",
                "household_debt_to_gdp_us",
                "cpi_us",
                "debt_credit_card_us",
                "inflation_us"
            ],
            "month_column": "date"
        },
        "Vietnam": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Vietnam_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/GB_country_vietnam.pkl",
            "features": [
                "wage_growth_us",
                "cpi_medical_care_services_us",
                "debt_credit_card_us",
                "cpi_energy_us",
                "cpi_commodities_less_food__energy_us"
            ],
            "month_column": "date"
        }
    },
    "hvc": {
        "Base": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Base_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/RF_hvc_base.pkl",
            "features": ["cpi_commodities_less_food__energy_us",
                         "pet_film_price_index",
                         "cotton_price",
                         "asp",
                         "mortgage_rate_us"]
        },
        "Embelex": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Embelex_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/GB_hvc_embelex.pkl",
            "features": ["crude_oil_wti_price",
                         "value_ft_turkey",
                         "pmi_vietnam",
                         "value_app_vietnam",
                         "pmi_turkey"]
        },
        "Ipps": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Ipps_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/RF_hvc_ipps.pkl",
            "features": ["pet_film_price_index",
                         "pmi_vietnam",
                         "cpi_commodities_less_food__energy_us",
                         "debt_home_equity_us",
                         "cotton_price"]
        },
        "Rfid": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Rfid_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/GB_hvc_rfid.pkl",
            "features": ["value_app_vietnam",
                         "value_ft_vietnam",
                         "pmi_vietnam",
                         "debt_credit_card_us",
                         "pmi_turkey"]
        }
    },
    "rbo": {
        "Adidas Grouped": {
            "features_excel": "",
            "simulation_model": "",
            "features": []
        },
        "Victoria's Secret Grouped": {
            "features_excel": "",
            "simulation_model": "",
            "features": []
        }
    },
    "top50_rbo": {
        "top50_rbo": {
            "features_excel": "dbfs:/FileStore/finalmlmodel/Feature List - Top_50_rbo_features.csv",
            "simulation_model": "dbfs:/FileStore/finalmlmodel/RF_Top50_model.pkl",
            "features": ["gdp_real_us",
                         "csi_us",
                         "cotton_price",
                         "csi_eu",
                         "total_weight_supplied_eur_china"]
        }
    },
    "settings": {
        "max_features": 5
    }
}


def get_max_tweakable_features():
    try:
        v = int(FILE_CONFIG.get('settings', {}).get('max_features', 5))
        if v < 1:
            return 1
        if v > 5:
            return 5
        return v
    except:
        return 5

def read_large_dbfs_file(file_path, chunk_size_mb=1):
    """Read large files from DBFS with proper chunking"""
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # First get file size
        status_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/get-status"
        status_response = requests.get(status_url, headers=headers, params={"path": file_path})
        
        if status_response.status_code != 200:
            st.error(f"❌ Cannot get file status: {status_response.status_code}")
            return None
        
        file_info = status_response.json()
        file_size = file_info.get('file_size', 0)
        
        st.info(f"📦 Reading {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # For very large files (>2MB), use special handling
        if file_size > 2 * 1024 * 1024:
            return read_very_large_file_via_multipart(file_path, file_size, chunk_size_mb)
        
        # For moderately large files, use standard chunked reading
        elif file_size > 500 * 1024:
            return read_chunked_file(file_path, file_size, chunk_size_kb=512)
        
        # For small files, read directly
        else:
            return read_small_file_directly(file_path)


    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None


def read_very_large_file_via_multipart(file_path, file_size, chunk_size_mb=1):
    """Read very large files using multipart approach"""
    chunk_size = chunk_size_mb * 1024 * 1024
    read_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/read"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    all_data = bytearray()
    offset = 0
    chunks_read = 0
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while offset < file_size:
        chunks_read += 1
        bytes_to_read = min(chunk_size, file_size - offset)
        
        status_text.text(f"📥 Reading chunk {chunks_read} ({offset:,} - {offset + bytes_to_read:,} bytes)")
        
        params = {
            "path": file_path,
            "offset": offset,
            "length": bytes_to_read
        }
        
        try:
            response = requests.get(read_url, headers=headers, params=params, timeout=60)
            
            if response.status_code != 200:
                st.error(f"❌ Failed to read chunk at offset {offset:,}: HTTP {response.status_code}")
                return None
            
            result = response.json()
            if 'data' not in result:
                st.error(f"❌ No data in chunk response at offset {offset:,}")
                return None
            
            chunk_data = base64.b64decode(result['data'])
            all_data.extend(chunk_data)
            offset += len(chunk_data)
            
            # Update progress
            progress = min(offset / file_size, 1.0)
            progress_bar.progress(progress)
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
            
        except requests.exceptions.Timeout:
            st.error(f"❌ Timeout reading chunk at offset {offset:,}")
            return None
        except Exception as e:
            st.error(f"❌ Error reading chunk at offset {offset:,}: {str(e)}")
            return None
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Successfully read {len(all_data):,} bytes in {chunks_read} chunks")
    time.sleep(0.5)  # Let user see the success message
    progress_bar.empty()
    status_text.empty()
    
    return bytes(all_data)


def read_chunked_file(file_path, file_size, chunk_size_kb=512):
    """Read moderately large files with chunking"""
    chunk_size = chunk_size_kb * 1024
    read_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/read"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    all_data = bytearray()
    offset = 0
    
    while offset < file_size:
        params = {
            "path": file_path,
            "offset": offset,
            "length": min(chunk_size, file_size - offset)
        }
        
        response = requests.get(read_url, headers=headers, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"❌ Failed to read at offset {offset:,}")
            return None
        
        result = response.json()
        if 'data' not in result:
            st.error(f"❌ No data in response at offset {offset:,}")
            return None
        
        chunk_data = base64.b64decode(result['data'])
        all_data.extend(chunk_data)
        offset += len(chunk_data)
    
    return bytes(all_data)


def read_small_file_directly(file_path):
    """Read small files directly"""
    read_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/read"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    params = {"path": file_path}
    response = requests.get(read_url, headers=headers, params=params, timeout=30)
    
    if response.status_code != 200:
        return None
    
    result = response.json()
    if 'data' in result:
        return base64.b64decode(result['data'])
    
    return None


def load_simulation_model_fixed(level, selected_filter=None):
    """Fixed model loading with special handling for large files"""
    try:
        # Get model path
        model_path = None
        if level == "country" and selected_filter in FILE_CONFIG["country"]:
            model_path = FILE_CONFIG["country"][selected_filter]["simulation_model"]
        elif level == "hvc" and selected_filter in FILE_CONFIG["hvc"]:
            model_path = FILE_CONFIG["hvc"][selected_filter]["simulation_model"]
        elif level == "rbo" and selected_filter in FILE_CONFIG["rbo"]:
            model_path = FILE_CONFIG["rbo"][selected_filter]["simulation_model"]
        elif level == "top50_rbo" and selected_filter in FILE_CONFIG["top50_rbo"]:
            model_path = FILE_CONFIG["top50_rbo"][selected_filter]["simulation_model"]
        
        if not model_path:
            st.error("❌ No model path configured")
            return None, [], []
        
        # st.info(f"🤖 Loading model: {model_path}")
        
        # Check file size first
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        status_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/get-status"
        status_response = requests.get(status_url, headers=headers, params={"path": model_path})
        
        if status_response.status_code != 200:
            st.error(f"❌ Cannot access model file")
            return None, [], []
        
        file_info = status_response.json()
        file_size = file_info.get('file_size', 0)
        
        # st.info(f"📊 Model file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Read the file with appropriate method based on size
        if file_size > 5 * 1024 * 1024:  # > 5MB
            st.warning("⚠️ Very large model file, this may take a moment...")
            model_content = read_very_large_file_via_multipart(model_path, file_size, chunk_size_mb=2)
        elif file_size > 500 * 1024:  # > 500KB
            model_content = read_chunked_file(model_path, file_size, chunk_size_kb=1024)
        else:
            model_content = read_small_file_directly(model_path)
        
        if model_content is None:
            st.error("❌ Failed to read model file")
            return None, [], []
        
        # Verify we got the full file
        if len(model_content) != file_size:
            st.warning(f"⚠️ File size mismatch: read {len(model_content):,} bytes, expected {file_size:,}")
            
            # For pickle files, sometimes we can still load them even if truncated
            # if they have padding or comments at the end
            if len(model_content) < file_size * 0.9:  # Less than 90% of expected size
                st.error("❌ File is significantly truncated")
                return None, [], []
        
        # Try to load pickle
        try:
            # First check if it's a valid pickle
            import pickle
            import io
            
            loaded_obj = pickle.load(io.BytesIO(model_content))
            # st.success(f"✅ Model loaded successfully: {type(loaded_obj).__name__}")
            
        except Exception as e:
            st.error(f"❌ Pickle loading failed: {str(e)}")
            
            # Try joblib for large models
            try:
                import joblib
                loaded_obj = joblib.load(io.BytesIO(model_content))
                st.success(f"✅ Model loaded with joblib: {type(loaded_obj).__name__}")
            except ImportError:
                st.warning("⚠️ joblib not available")
            except Exception as e2:
                st.error(f"❌ Joblib also failed: {str(e2)}")
                return None, [], []
        
        # Extract features
        all_features = []
        tweakable_features = []
        
        if isinstance(loaded_obj, dict):
            # Look for features
            for key in ['selected_features', 'feature_names', 'features', 'columns']:
                if key in loaded_obj:
                    features = loaded_obj[key]
                    if isinstance(features, (list, np.ndarray)):
                        all_features = list(features)
                        break
        
        # If no features found, use config
        if not all_features:
            all_features = get_features_for_level(level, selected_filter)
        
        # Get tweakable features
        tweakable_features = get_features_for_level(level, selected_filter)
        
        return loaded_obj, all_features, tweakable_features
        
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, [], []


# Update your FILE_CONFIG to include model sizes for reference
FILE_CONFIG["model_sizes"] = {
    "RF_Top50_model.pkl": 7319248,
    "GB_hvc_embelex.pkl": 690400,
    "GB_hvc_rfid.pkl": 386090,
    "RF_hvc_rfid.pkl": 1067594,
    "RF_country_china.pkl": 459032,
    "XGB_country_honduras.pkl": 458030,
    "GB_country_turkey.pkl": 420093,
    "XGB_country_bangladesh.pkl": 402428,
    "GB_country_vietnam.pkl": 268470,
    "RF_hvc_ipps.pkl": 277337,
    "RF_hvc_base.pkl": 195288
}


# Add a preload option to handle large models
def preload_large_models():
    """Preload large models in background"""
    if 'large_models_loaded' not in st.session_state:
        st.session_state.large_models_loaded = {}
    
    # List of large models to preload
    large_models = [
        ("top50_rbo", "top50_rbo", "RF_Top50_model.pkl"),
        ("hvc", "Embelex", "GB_hvc_embelex.pkl"),
        ("hvc", "Rfid", "GB_hvc_rfid.pkl"),
        ("hvc", "Rfid", "RF_hvc_rfid.pkl")  # Note: two RFID models?
    ]
    
    for level, filter_name, model_name in large_models:
        cache_key = f"{level}_{filter_name}"
        
        if cache_key not in st.session_state.large_models_loaded:
            # st.info(f"🔄 Preloading {model_name}...")
            
            # Get model path from config
            model_path = None
            if level == "top50_rbo":
                model_path = FILE_CONFIG["top50_rbo"]["top50_rbo"]["simulation_model"]
            elif level == "hvc" and filter_name in FILE_CONFIG["hvc"]:
                model_path = FILE_CONFIG["hvc"][filter_name]["simulation_model"]
            
            if model_path:
                try:
                    # Load in background thread
                    import threading
                    
                    def load_model_thread():
                        model, features, tweakable = load_simulation_model_fixed(level, filter_name)
                        if model is not None:
                            st.session_state.large_models_loaded[cache_key] = (model, features, tweakable)
                    
                    thread = threading.Thread(target=load_model_thread)
                    thread.daemon = True
                    thread.start()
                    
                except:
                    pass


# Call this at the start of your app
preload_large_models()

def load_top50_model_special():
    """Special handling for the 7.3MB Top50 model"""
    model_path = "dbfs:/FileStore/finalmlmodel/RF_Top50_model.pkl"
    
    st.toast("⚠️ Loading large Top50 model (7.3MB), please be patient...")
    
    # Try to download in parts
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read in 1MB chunks
        chunk_size = 1024 * 1024  # 1MB
        all_data = bytearray()
        offset = 0
        
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # First get file size
        status_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/get-status"
        status_response = requests.get(status_url, headers=headers, params={"path": model_path})
        
        if status_response.status_code != 200:
            st.error("❌ Cannot get file size")
            return None, [], []
        
        file_size = status_response.json().get('file_size', 0)
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        
        # Read all chunks
        for chunk_num in range(total_chunks):
            status_text.text(f"📥 Downloading chunk {chunk_num + 1}/{total_chunks}")
            
            read_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/read"
            params = {
                "path": model_path,
                "offset": offset,
                "length": min(chunk_size, file_size - offset)
            }
            
            response = requests.get(read_url, headers=headers, params=params, timeout=60)
            
            if response.status_code != 200:
                st.error(f"❌ Failed to download chunk {chunk_num + 1}")
                return None, [], []
            
            result = response.json()
            if 'data' not in result:
                st.error(f"❌ No data in chunk {chunk_num + 1}")
                return None, [], []
            
            chunk_data = base64.b64decode(result['data'])
            all_data.extend(chunk_data)
            offset += len(chunk_data)
            
            # Update progress
            progress = (chunk_num + 1) / total_chunks
            progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Download complete")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Now load the pickle
        import io
        import pickle
        
        model_content = bytes(all_data)
        
        # Try to load
        try:
            model = pickle.load(io.BytesIO(model_content))
            st.success("✅ Top50 model loaded successfully")
            
            # Extract features
            all_features = []
            if hasattr(model, 'feature_names_in_'):
                all_features = list(model.feature_names_in_)
            
            # Get tweakable features from config
            tweakable_features = get_features_for_level("top50_rbo", "top50_rbo")
            
            return model, all_features, tweakable_features
            
        except Exception as e:
            st.error(f"❌ Failed to load pickle: {str(e)}")
            return None, [], []
            
    except Exception as e:
        st.error(f"❌ Error loading Top50 model: {str(e)}")
        return None, [], []




# Replace your existing load_simulation_model function with this:
def load_simulation_model(level, selected_filter=None):
    """Main model loading function with fallbacks"""
    # First check cache
    cache_key = f"{level}_{selected_filter}"
    if cache_key in st.session_state.get('large_models_loaded', {}):
        return st.session_state.large_models_loaded[cache_key]
    
    if level == "top50_rbo" and selected_filter == "top50_rbo":
        return load_top50_model_special()
        
    # Otherwise load fresh
    return load_simulation_model_fixed(level, selected_filter)


# =============================================================================
# DATABRICKS FILE OPERATIONS
# =============================================================================
def read_dbfs_file_via_api(file_path):
    """Read file from DBFS using Databricks API"""
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        read_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/read"
        params = {"path": file_path}
        response = requests.get(read_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if 'data' in result:
                file_content = base64.b64decode(result['data'])
                return file_content
            else:
                return None
        else:
            return None
    except Exception as e:
        return None


def check_dbfs_access():
    """Check if we can access DBFS and list available files"""
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        list_url = f"{DATABRICKS_WORKSPACE_URL}/api/2.0/dbfs/list"
        params = {"path": "dbfs:/FileStore/finalmlmodel/"}
        response = requests.get(list_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if 'files' in result:
                st.toast("✅ Successfully connected to DBFS")
                # for file_info in result['files']:
                    # st.toast(f"  - {file_info['path']} (Size: {file_info['file_size']} bytes)")
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        return False


def initialize_file_access():
    """Initialize and verify file access"""
    st.toast("🔍 Checking Databricks file access...")
    if check_dbfs_access():
        st.toast("✅ Databricks file access configured correctly")
        return True
    else:
        st.error("❌ Cannot access Databricks files. Please check configuration.")
        return False


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_features_excel(level, selected_filter):
    """Load features and their values from Excel file"""
    try:
        if level == "country" and selected_filter in FILE_CONFIG["country"]:
            excel_path = FILE_CONFIG["country"][selected_filter]["features_excel"]
        elif level == "hvc" and selected_filter in FILE_CONFIG["hvc"]:
            excel_path = FILE_CONFIG["hvc"][selected_filter]["features_excel"]
        elif level == "rbo" and selected_filter in FILE_CONFIG["rbo"]:
            excel_path = FILE_CONFIG["rbo"][selected_filter]["features_excel"]
        elif level == "top50_rbo" and selected_filter in FILE_CONFIG["top50_rbo"]:
            excel_path = FILE_CONFIG["top50_rbo"][selected_filter]["features_excel"]
        else:
            return pd.DataFrame()
        
        if not excel_path:
            return pd.DataFrame()
        
        file_content = read_dbfs_file_via_api(excel_path)
        if file_content is None:
            return pd.DataFrame()
            
        excel_file = io.BytesIO(file_content)
        df = pd.read_csv(excel_file)
        
        if df is None or df.empty:
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        return pd.DataFrame()


# =============================================================================
# FEATURE NAME MAPPING
# =============================================================================
FEATURE_MAPPING = {
    "inflation_us": "Consumer Price Index for All Urban Consumers",
    "asp": "Average Selling Price",
    "cpi_commodities_less_food__energy_us": "CPI Commodities Less Food & Energy (US)",
    "cpi_apparel_us": "CPI Apparel (US)",
    "debt_credit_card_us": "Credit Card Debt (US)",
    "cpi_services_less_energy_services_us": "CPI Services Less Energy Services (US)",
    "pet_film_price_index": "PET Film Price Index",
    "wage_growth_eu": "Wage Growth (EU)",
    "debt_auto_loan_us": "Auto Loan Debt (US)",
    "total_weight_supplied_us_bangladesh": "Total Weight Supplied US-Bangladesh",
    "housing_starts_singlefamily_units_us": "Housing Starts Single Family Units (US)",
    "wage_growth_us": "Wage Growth (US)",
    "gdp_real_eu": "Real GDP (EU)",
    "pmi_honduras": "PMI Honduras",
    "rfid_ic_price": "RFID IC Price",
    "unemployment_eu": "Unemployment (EU)",
    "household_debt_to_gdp_us": "Household Debt to GDP (US)",
    "cpi_us": "Consumer Price Index (US)",
    "inflation_us": "Inflation (US)",
    "cpi_medical_care_services_us": "CPI Medical Care Services (US)",
    "cpi_energy_us": "CPI Energy (US)",
}


def get_business_name(feature_name: str) -> str:
    """Translates a technical feature name to a business name."""
    return FEATURE_MAPPING.get(feature_name, feature_name.replace('_', ' ').title())


def get_top_features_for_level(level, selected_filter, max_features: Optional[int] = None):
    """Get top features for a given level and filter (returns dataframe)."""
    try:
        features = get_features_for_level(level, selected_filter)
        if not features:
            return pd.DataFrame(columns=["feature", "importance", "importance_numeric", "importance_display"])
        
        if max_features is None:
            max_features = get_max_tweakable_features()
        
        features_to_return = features[:max_features]
        importance_numeric = [round(100.0 / max_features * (max_features - i), 2) for i in range(len(features_to_return))]
        importance_display = [f"{int(v)}%" for v in importance_numeric]
        
        df = pd.DataFrame({
            "feature": features_to_return,
            "importance": importance_display,
            "importance_numeric": importance_numeric,
            "importance_display": importance_display
        })
        
        df['feature_business_name'] = df['feature'].apply(get_business_name)
        return df
        
    except Exception as e:
        return pd.DataFrame()


def get_features_for_level(level, selected_filter):
    """Get features for a given level and filter from FILE_CONFIG"""
    try:
        if level in FILE_CONFIG and selected_filter in FILE_CONFIG[level]:
            features = FILE_CONFIG[level][selected_filter].get("features", [])
            return features
        else:
            return []
    except Exception as e:
        return []


# =============================================================================
# FIXED MODEL LOADING
# =============================================================================




def make_prediction(model, input_array):
    """Universal prediction function"""
    try:
        if isinstance(model, dict):
            for key, value in model.items():
                if hasattr(value, 'predict'):
                    return value.predict(input_array)[0]
            
            model_keys_to_check = ['model', 'estimator', 'regressor', 'classifier', 'best_estimator', 'pipeline']
            for key in model_keys_to_check:
                if key in model and hasattr(model[key], 'predict'):
                    return model[key].predict(input_array)[0]
            
            for key, value in model.items():
                if callable(getattr(value, 'predict', None)):
                    return value.predict(input_array)[0]
            
            return 0.0
        
        if isinstance(model, Pipeline):
            if hasattr(model, 'predict'):
                return model.predict(input_array)[0]
            else:
                return 0.0
        
        if hasattr(model, 'predict'):
            return model.predict(input_array)[0]
        
        return 0.0
                
    except Exception as e:
        return 0.0


# =============================================================================
# CURRENT FORECAST FUNCTION
# =============================================================================

def get_current_forecast_from_model(level, selected_filter, forecast_month):

    try:
        # ------------------------------------------------------------------
        # 1. Load Model & Features
        # ------------------------------------------------------------------
        model, all_features, tweakable_features = load_simulation_model(level, selected_filter)
        
        if model is None:
            # st.error("Model not found.") # Optional: Un-comment for UI debugging
            return 0.0

        # ------------------------------------------------------------------
        # 2. Get Input Data (The Feature Values from Excel)
        # ------------------------------------------------------------------
        # matches: baseline_inputs_dict = get_baseline_feature_values(...)
        baseline_inputs_dict = get_baseline_feature_values(level, selected_filter, all_features, forecast_month)

        if not baseline_inputs_dict:
            # st.warning(f"No data found for {forecast_month}.") # Optional
            return 0.0

        # ------------------------------------------------------------------
        # 3. Prepare Data for Model (Align with 'all_features')
        # ------------------------------------------------------------------
        # matches: baseline_data_list = [baseline_inputs_dict.get(f, 0.0) for f in all_features]
        baseline_data_list = [baseline_inputs_dict.get(f, 0.0) for f in all_features]
        
        # matches: baseline_array = np.array(baseline_data_list).reshape(1, -1)
        baseline_array = np.array(baseline_data_list).reshape(1, -1)

        # ------------------------------------------------------------------
        # 4. Make Prediction
        # ------------------------------------------------------------------
        # matches: current_forecast = float(make_prediction(model, baseline_array))
        current_forecast = float(make_prediction(model, baseline_array))
        
        return current_forecast

    except Exception as e:
        # Fallback for any unexpected errors
        print(f"Error in get_current_forecast_from_model: {e}")
        return 0.0


# =============================================================================
# FIXED BASELINE FEATURE VALUES FUNCTION
# =============================================================================

def get_baseline_feature_values(
    level: str, 
    selected_filter: str, 
    all_features: List[str], 
    selected_month_str: str  # e.g., "October 2025"
) -> Dict[str, float]:
    """
    Get baseline values filtering by both MONTH and YEAR.
    """
    # 1. Safe Fallback
    def get_empty_response():
        return {feature: 0.0 for feature in all_features}

    try:
        # 2. Load Data
        features_df = load_features_excel(level, selected_filter)
        if features_df.empty:
            return get_empty_response()

        # 3. Parse User Selection (e.g., "October 2025")
        try:
            # Parses "October 2025" into a datetime object
            target_date = parser.parse(selected_month_str)
            target_year = target_date.year   # 2025
            target_month = target_date.month # 10
        except (ValueError, TypeError):
            st.error(f"❌ Could not parse selected date: '{selected_month_str}'")
            return get_empty_response()

        # 4. Find the Date Column
        # We look for columns containing 'date' or 'month' (case-insensitive)
        date_col = next((col for col in features_df.columns 
                         if 'date' in col.lower() or 'month' in col.lower()), None)
        
        row_data = pd.Series(dtype=float)

        if date_col:
            # CRITICAL STEP: Convert Excel column to Datetime objects
            # This handles strings like "01-10-2025" or "2025-10-01" automatically
            features_df[date_col] = pd.to_datetime(features_df[date_col], errors='coerce')

            # 5. Filter by BOTH Year and Month
            mask = (
                (features_df[date_col].dt.year == target_year) & 
                (features_df[date_col].dt.month == target_month)
            )
            
            filtered_data = features_df[mask]
            
            if not filtered_data.empty:
                # Take the first match found for that month/year
                row_data = filtered_data.iloc[0]
            else:
                st.warning(f"⚠️ Data loaded, but no row found specifically for {selected_month_str}. Using first available row as fallback.")
                row_data = features_df.iloc[0]
        else:
            st.warning("⚠️ Could not identify a 'Date' column in the Excel file.")
            row_data = features_df.iloc[0]

        # 6. Extract and Clean Data (Vectorized)
        subset = row_data.reindex(all_features)
        cleaned_values = pd.to_numeric(subset, errors='coerce').fillna(0.0)

        return cleaned_values.to_dict()

    except Exception as e:
        st.error(f"❌ Error retrieving baseline features: {str(e)}")
        return get_empty_response()





# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def generate_simulation(level, selected_filter, feature_adjustments, selected_month, sim_key):
    """
    Generates simulation, calculates Baseline vs Simulated from the PKL, 
    and DISPLAYS results immediately.
    """
    # print(f"DEBUG: Starting simulation for {selected_month}...") # Console debug

    try:
        # 1. Load Model & Features
        model, all_features, tweakable_features = load_simulation_model(level, selected_filter)
        if model is None:
            st.error("Model not found.")
            return pd.DataFrame()

        # 2. Get Input Data (The Feature Values from Excel)
        baseline_inputs_dict = get_baseline_feature_values(level, selected_filter, all_features, selected_month)
        
        if not baseline_inputs_dict:
            st.warning(f"No data found for {selected_month}. Check Excel file dates.")
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # STEP A: Calculate BASELINE Forecast (From Model)
        # ------------------------------------------------------------------
        baseline_data_list = [baseline_inputs_dict.get(f, 0.0) for f in all_features]
        baseline_array = np.array(baseline_data_list).reshape(1, -1)
        
        # This is the "Current Forecast" according to the Model
        current_forecast = float(make_prediction(model, baseline_array))

        # ------------------------------------------------------------------
        # STEP B: Calculate SIMULATED Forecast (From Model + Adjustments)
        # ------------------------------------------------------------------
        adjusted_inputs_dict = baseline_inputs_dict.copy()
        
        # Apply % changes
        for feature in tweakable_features:
            if feature in feature_adjustments:
                original_val = baseline_inputs_dict.get(feature, 0.0)
                pct_change = feature_adjustments[feature]
                # New = Old + (Old * %)
                adjusted_inputs_dict[feature] = original_val + (original_val * (pct_change / 100.0))

        sim_data_list = [adjusted_inputs_dict.get(f, 0.0) for f in all_features]
        sim_array = np.array(sim_data_list).reshape(1, -1)
        
        simulated_prediction = float(make_prediction(model, sim_array))

        # ------------------------------------------------------------------
        # STEP C: Calculate Difference & VISUALIZE (Restore UI)
        # ------------------------------------------------------------------
        abs_diff = simulated_prediction - current_forecast
        pct_diff = (abs_diff / current_forecast * 100) if current_forecast != 0 else 0.0

        # --- VISUAL 1: Display Metrics Immediately (Like earlier) ---
        st.markdown("### 📊 Simulation Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Baseline Forecast", f"{current_forecast:,.0f}")
        with col2:
            st.metric("Simulated Forecast", f"{simulated_prediction:,.0f}", delta=f"{pct_diff:.2f}%")
        with col3:
            st.metric("Difference", f"{abs_diff:+,.0f}")

        # --- VISUAL 2: Session State Success Message ---
        success_msg = (
            f"🎯 **Simulation complete!**\n"
            f"**{selected_filter} ({selected_month})**\n"
            f"New: {simulated_prediction:,.2f} | Baseline: {current_forecast:,.2f} | Diff: {abs_diff:+,.2f}"
        )
        
        if sim_key not in st.session_state:
            st.session_state[sim_key] = {}
        st.session_state[sim_key]['success_message'] = success_msg
        st.session_state.show_debug = True
        
        # Display the blue info box if your app relies on it
        st.info(success_msg) 

        # ------------------------------------------------------------------
        # STEP D: Return DataFrame (For Charts/Downstream Code)
        # ------------------------------------------------------------------
        
        # Date Parsing for the DataFrame
        try:
            date_obj = pd.to_datetime(selected_month, format='%B %Y', errors='coerce')
            if pd.isna(date_obj): date_obj = pd.to_datetime(selected_month, errors='coerce')
            if pd.isna(date_obj): date_obj = pd.to_datetime("2025-10-01") # Fallback
        except:
            date_obj = pd.to_datetime("2025-10-01")

        result_df = pd.DataFrame({
            'date': [date_obj],
            'simulated_forecast': [simulated_prediction],
            'current_forecast_1': [current_forecast],
            'difference': [abs_diff],
            'Year': [date_obj.year],
            'Month': [date_obj.month]
        })

        return result_df

    except Exception as e:
        st.error(f"Simulation Error: {e}")
        # print(f"Detailed Error: {e}")
        return pd.DataFrame()

def generate_baseline_forecast(level, selected_filter, forecast_months=3):
    """Generate baseline forecast using the model"""
    try:
        forecast_data = []
        forecast_months_list = ["October 2025", "November 2025", "December 2025"][:forecast_months]
        
        for month in forecast_months_list:
            forecast_value = get_current_forecast_from_model(level, selected_filter, month)
            
            month_mapping = {
                "October 2025": pd.Timestamp("2025-10-01"),
                "November 2025": pd.Timestamp("2025-11-01"),
                "December 2025": pd.Timestamp("2025-12-01")
            }
            
            forecast_data.append({
                'date': month_mapping[month],
                'actual_forecast': forecast_value,
                'Year': 2025,
                'Month': month_mapping[month].month,
                'historical_sales': 0
            })
        
        future_data = pd.DataFrame(forecast_data)
        return future_data
        
    except Exception as e:
        return pd.DataFrame()


def generate_future_forecast(historical_data, level_filters, level, selected_filter, forecast_months=3):
    """Generate future forecast"""
    if historical_data.empty:
        return pd.DataFrame()
    
    if level in ["country", "hvc", "rbo", "top50_rbo"] and selected_filter is not None:
        return generate_baseline_forecast(level, selected_filter, forecast_months)
    
    return pd.DataFrame()


def generate_simulated_forecast(historical_data, level_filters, level, selected_filter, exog_adjustments, sim_month, sim_key):
    """Generate simulated forecast for a specific month"""
    try:
        simulated_data = generate_simulation(level, selected_filter, exog_adjustments, sim_month, sim_key)
        return simulated_data
    except Exception as e:
        return pd.DataFrame()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_forecast_plot(
    historical_data,
    forecast_data,
    selected_years=None,
    show_simulated=False,
    level_filters=None,
    level=None,
    selected_filter=None
):
    """
    Create a plot with historical sales and future forecast.

    Args:
        historical_data (pd.DataFrame): Historical sales data.
        forecast_data (pd.DataFrame): Forecast data with optional simulated forecast.
        selected_years (list): List of years to filter historical data (as int or str).
        show_simulated (bool): Whether to show simulated forecast.
        level_filters (dict): Dict containing filter options for country, RBO, HVC, top50_rbo.
        level (str): Level of filtering ('country', 'rbo', 'hvc', 'top50_rbo').
        selected_filter (str): Selected value for level filter.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object.
    """

    # Check if both datasets are empty
    if (historical_data is None or historical_data.empty) and (forecast_data is None or forecast_data.empty):
        st.warning("No data available to plot.")
        return None

    # Copy historical data
    filtered_historical = historical_data.copy()

    # Convert columns to lowercase for safe filtering
    for col in ["country", "rbo", "HVC", "top50_rbo"]:
        if col in filtered_historical.columns:
            filtered_historical[col] = filtered_historical[col].astype(str).str.lower()
    if selected_filter:
        selected_filter = selected_filter.lower()

    # Apply level filters
    if level_filters and level and selected_filter:
        level_map = {
            "country": "country",
            "rbo": "rbo",
            "hvc": "HVC",
            "top50_rbo": "top50_rbo"
        }
        col_name = level_map.get(level)
        if col_name in filtered_historical.columns:
            filtered_historical = filtered_historical[filtered_historical[col_name] == selected_filter]

    # Ensure 'date' column exists and is datetime
    if "date" not in filtered_historical.columns or filtered_historical["date"].isnull().all():
        filtered_historical["date"] = pd.to_datetime(filtered_historical["Date"], errors="coerce")

    # Ensure 'historical_sales' is numeric
    if "historical_sales" in filtered_historical.columns:
        filtered_historical["historical_sales"] = pd.to_numeric(filtered_historical["historical_sales"], errors="coerce")

    # Handle year filter
    if selected_years:
        filtered_historical["Year"] = filtered_historical["Year"].astype(str)
        selected_years = [str(y) for y in selected_years]
        filtered_historical = filtered_historical[filtered_historical["Year"].isin(selected_years)]

    # Drop rows with missing date or sales
    filtered_historical = filtered_historical.dropna(subset=["date", "historical_sales"])

    # Create the figure
    fig = px.line(title="Actual Forecast Vs Simulated Forecast")

    # Add historical sales line
    if not filtered_historical.empty:
        historical_grouped = filtered_historical.groupby("date", as_index=False)["historical_sales"].sum()
        fig.add_scatter(
            x=historical_grouped["date"],
            y=historical_grouped["historical_sales"],
            mode="lines",
            name="Historical Sales",
            line=dict(color="green", width=2, dash="dot"),
            connectgaps=True
        )
    else:
        st.info("No historical sales data available for the selected filters/years.")

    # Add current forecast
    if forecast_data is not None and not forecast_data.empty and "actual_forecast" in forecast_data.columns:
        forecast_data = forecast_data.dropna(subset=["date", "actual_forecast"])
        fig.add_scatter(
            x=forecast_data["date"],
            y=forecast_data["actual_forecast"],
            mode="lines+markers",
            name="Current Forecast (2025)",
            line=dict(color="#FF4500", width=3),
            connectgaps=True
        )

    # Add simulated forecast if requested
    if show_simulated and forecast_data is not None and not forecast_data.empty and "simulated_forecast" in forecast_data.columns:
        forecast_data = forecast_data.dropna(subset=["date", "simulated_forecast"])
        fig.add_scatter(
            x=forecast_data["date"],
            y=forecast_data["simulated_forecast"],
            mode="lines+markers",
            name="Simulated Forecast (2025)",
            line=dict(color="blue", width=3, dash="dash"),
            connectgaps=True
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales Value ($)",
        hovermode="x unified",
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    

    return fig





def create_comparison_table(forecast_data):
    """Create comparison table between actual and simulated forecast"""
    if 'simulated_forecast' not in forecast_data.columns:
        return None
    
    comparison_data = forecast_data[['date', 'actual_forecast', 'simulated_forecast']].copy()
    comparison_data['difference'] = comparison_data['simulated_forecast'] - comparison_data['actual_forecast']
    comparison_data['percentage_change'] = (comparison_data['difference'] / comparison_data['actual_forecast']) * 100
    
    comparison_data_display = comparison_data.copy()
    comparison_data_display['date'] = comparison_data_display['date'].dt.strftime('%Y-%m-%d')
    
    for col in ['actual_forecast', 'simulated_forecast', 'difference']:
        comparison_data_display[col] = comparison_data_display[col].round(2)
    
    comparison_data_display['percentage_change'] = comparison_data_display['percentage_change'].round(2)
    
    comparison_data_display = comparison_data_display.rename(columns={
        'date': 'Date',
        'actual_forecast': 'Current Forecast',
        'simulated_forecast': 'Simulated Forecast',
        'difference': 'Difference',
        'percentage_change': 'Change %'
    })
    
    return comparison_data_display


# =============================================================================
# FORECAST TAB CREATION (FROM ORIGINAL UI)
# =============================================================================


def get_historical_years(data):
    if data.empty or 'Year' not in data.columns:
        return []
    return sorted([str(y) for y in data['Year'].unique()])



def count_combinations(data, level_filters, level):
    """Count the number of unique combinations based on selected filters for each level"""
    if data.empty:
        return 0
    
    return 1


def get_top_features(level, selected_filter, top_n=5):
    """Get top N features by importance and include their business names."""
    try:
        feature_importance_df = get_top_features_for_level(level, selected_filter, max_features=get_max_tweakable_features())
        if feature_importance_df.empty:
            return pd.DataFrame()
        
        return feature_importance_df.head(top_n)
    except Exception as e:
        return pd.DataFrame()


def create_level_forecast_tab(historical_data, level, level_filters, selected_filter):
    """Create forecast content for each level tab"""
    
    sim_key = f"sim_{level}_{selected_filter}"
    
    if not selected_filter and level in ["country", "hvc", "rbo", "top50_rbo"]:
        st.warning(f"⚠️ Please select a {level} to generate forecasts")
        return

    historical_years = get_historical_years(historical_data)
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)

    combination_count = count_combinations(historical_data, level_filters, level)

    if level in ["country", "hvc", "rbo", "top50_rbo"] and combination_count > 0:
        sim_key = f"sim_{level}_{selected_filter}"
        
        # Initialize Session State
        if sim_key not in st.session_state:
            st.session_state[sim_key] = {
                'data': None,
                'adjustments': {},
                'generated': False,
                'reset_count': 0,
                'selected_month': "October 2025",
                'simulation_months': ["October 2025", "November 2025", "December 2025"],
                'simulation_results': None
            }

        # Initial Load of Standard Forecast (This might be from Excel/Cache initially)
        with st.spinner(f"Generating 2025 forecast using {level} model..."):
            base_forecast_data = generate_future_forecast(
                historical_data, level_filters, level, selected_filter, forecast_months=3
            )

        if base_forecast_data.empty:
            st.warning("No forecast data available for the selected filters.")
            return

        # ==============================================================================
        # MAIN LAYOUT
        # ==============================================================================
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.subheader("📈 Sales Metrics")

            simulation_month_options = ["October 2025", "November 2025", "December 2025"]
            current_simulation_months = st.session_state.get(sim_key, {}).get('simulation_months', simulation_month_options)
            
            simulation_months = st.multiselect(
                "Select months for simulation:",
                options=simulation_month_options,
                default=current_simulation_months,
                key=f"{level}_{selected_filter}_simulation_months"
            )
            st.session_state[sim_key]['simulation_months'] = simulation_months
            
            # --- METRIC CALCULATION LOGIC ---
            # Default to base data
            total_base_forecast_all_months = base_forecast_data['actual_forecast'].sum() if not base_forecast_data.empty and 'actual_forecast' in base_forecast_data.columns else 0
            
            month_mapping = {"October 2025": 10, "November 2025": 11, "December 2025": 12}
            simulation_month_nums = [month_mapping[month] for month in simulation_months if month in month_mapping]

            # If simulation has run, use the UPDATED data (which has PKL values for both)
            if st.session_state[sim_key]['generated'] and st.session_state[sim_key]['data'] is not None:
                simulation_data = st.session_state[sim_key]['data']
                filtered_simulation_data = simulation_data[simulation_data['Month'].isin(simulation_month_nums)]
                
                # Get Simulated Total
                total_simulated = filtered_simulation_data['simulated_forecast'].sum() if not filtered_simulation_data.empty else 0
                simulated_display = f"${total_simulated:,.0f}"
                
                # Get Baseline Total (Crucial: This now comes from 'actual_forecast' which we updated with PKL values)
                total_forecast = filtered_simulation_data['actual_forecast'].sum() if not filtered_simulation_data.empty else 0
                
                forecast_subtitle = f"{', '.join(simulation_months)}"
            else:
                # Default view before simulation
                if not simulation_months:
                    forecast_subtitle = "Baseline (no months selected)"
                    simulated_display = "--"
                    total_forecast = 0
                else:
                    simulated_display = "--"
                    filtered_base_data = base_forecast_data[base_forecast_data['Month'].isin(simulation_month_nums)]
                    total_forecast = filtered_base_data['actual_forecast'].sum() if not filtered_base_data.empty else 0
                    forecast_subtitle = "[Oct-Dec]"

            st.markdown(f'''
                <div style="display: flex; justify-content: space-between; gap: 10px;">
                    <div style="flex: 1;">
                        <div class="metric-card">
                            <div class="metric-title">Simulated Sales of {simulation_months}</div>
                            <div class="metric-value">{simulated_display}</div>
                            <div class="metric-subtitle">After adjustments</div>
                        </div>
                    </div>
                    <div style="flex: 1;">
                        <div class="metric-card">
                            <div class="metric-title">Forecasted Sales of {forecast_subtitle}</div>
                            <div class="metric-value">${total_forecast:,.0f}</div>
                            <div class="metric-subtitle">{forecast_subtitle}</div>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; min-height: 48px; margin-bottom: 1rem;">
                    <h2 style="font-size: 1.5rem; font-weight: 600; margin: 0; padding: 0;">📊 Top Exogeneous Features</h2>
                    <span title="• These Features represent Only those which Can be tweaked for Model Simulation • Time based features are not included;">ℹ️</span>
                </div>
            """, unsafe_allow_html=True)

            top_features_df = get_top_features(level, selected_filter)
            
            if not top_features_df.empty:
                for _, row in top_features_df.iterrows():
                    feature_display = row['feature_business_name']
                    st.markdown(f'''
                        <div class="feature-item">
                            <span class="feature-name" style=" font-size: 15px;">{feature_display}</span>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No feature importance data available for this model.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ==============================================================================
        # FEATURE ADJUSTMENT SECTION
        # ==============================================================================
        st.markdown("---")
        st.subheader("⚙️ Feature Adjustments")

        col_month, col_info = st.columns([1, 2])

        with col_month:
            month_options = simulation_months if simulation_months else ["October 2025"]
            current_selected_month = st.session_state.get(sim_key, {}).get('selected_month', month_options[0] if month_options else "October 2025")
            
            if current_selected_month in month_options:
                default_index = month_options.index(current_selected_month)
            else:
                default_index = 0
                
            selected_month = st.selectbox(
                "Select Month for Feature Adjustment:",
                options=month_options,
                index=default_index,
                key=f"{level}_{selected_filter}_month_selector"
            )
            st.session_state[sim_key]['selected_month'] = selected_month
            
            # Get baseline feature values for the selected simulation month
            if not top_features_df.empty:
                features_to_fetch = top_features_df['feature'].tolist()
                baseline_feature_values = get_baseline_feature_values(
                    level, selected_filter, features_to_fetch, selected_month
                )
            else:
                baseline_feature_values = {}

        # with col_info:
            # st.info(f"🔹 **Adjustment Month** - Feature baseline values below are for **{selected_month}**.")

        if 'max_features' not in st.session_state:
            st.session_state['max_features'] = get_max_tweakable_features()
            
        MAX_FEATURE_SLIDERS = st.session_state.get('max_features', get_max_tweakable_features())

        exog_adjustments = {}
        if not top_features_df.empty:
            top_features_df = top_features_df.head(MAX_FEATURE_SLIDERS)
            num_features = len(top_features_df)
            num_cols = min(3, num_features) if num_features > 0 else 1
            
            cols = st.columns(num_cols)
            for i, (_, row) in enumerate(top_features_df.iterrows()):
                feature_name = row['feature']
                feature_display = row['feature_business_name']
                slider_key = f"{level}_{selected_filter}_{feature_name}_slider_{st.session_state[sim_key]['reset_count']}"
                
                default_value = st.session_state[sim_key]['adjustments'].get(feature_name, 0)
                baseline_value = baseline_feature_values.get(feature_name, 0.0)
                
                if baseline_value > 1000 or baseline_value < -1000:
                    baseline_display = f"{baseline_value:,.2f}"
                else:
                    baseline_display = f"{baseline_value:,.4f}"
                
                selected_col = cols[i % num_cols]
                selected_col.markdown(
                    f"""
                    <div style='padding-top: 10px; font-size: 13px; color: #333;'>
                        <strong>{feature_display}</strong> | Baseline: <strong>{baseline_display}</strong>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                percentage_change = selected_col.slider(
                    "Adjustment (%)",
                    min_value=-50,
                    max_value=50,
                    value=default_value,
                    step=5,
                    format="%d%%",
                    key=slider_key,
                    label_visibility='collapsed',
                    help=f"Current baseline value for {selected_month} is {baseline_display}. Adjust {feature_display} by percentage."
                )
                exog_adjustments[feature_name] = percentage_change

        button_col1, button_col2 = st.columns([1, 1])
        with button_col1:
            generate_key = f"generate_{level}_{selected_filter}_{st.session_state[sim_key]['reset_count']}"
            generate_forecast = st.button(
                "🚀 Generate Simulated Forecast", use_container_width=True, type="primary", key=generate_key
            )
        with button_col2:
            reset_key = f"reset_{level}_{selected_filter}_{st.session_state[sim_key]['reset_count']}"
            reset_simulation = st.button(
                "🔄 Reset Simulation", use_container_width=True, key=reset_key
            )

        if reset_simulation:
            st.session_state[sim_key]['reset_count'] += 1
            st.session_state[sim_key]['data'] = None
            st.session_state[sim_key]['adjustments'] = {}
            st.session_state[sim_key]['generated'] = False
            st.session_state[sim_key]['selected_month'] = "October 2025"
            st.session_state[sim_key]['simulation_months'] = ["October 2025", "November 2025", "December 2025"]
            st.session_state[sim_key]['simulation_results'] = None
            st.session_state.show_debug = False
            st.success("✅ Simulation reset successfully! Showing base forecast.")
            st.rerun()

        adjustments_made = any(adj != 0 for adj in exog_adjustments.values()) if exog_adjustments else False

        # ==============================================================================
        # GENERATE FORECAST LOGIC (Crucial Updates Here)
        # ==============================================================================
        if generate_forecast:
            st.session_state[sim_key]['simulation_results'] = None
            
            with st.spinner(f"🔄 Generating simulated forecast using real-time Model..."):
                if base_forecast_data.empty:
                    st.warning("No forecast data available for the selected filters.")
                    return
                
                if adjustments_made:
                    st.session_state[sim_key]['adjustments'] = exog_adjustments
                    
                    merged_data = base_forecast_data.copy()
                    merged_data['simulated_forecast'] = np.nan
                    
                    simulation_results = []
                    
                    # Iterate through all selected months
                    for sim_month in st.session_state[sim_key]['simulation_months']:
                        
                        # --- CALL THE ROBUST PKL FUNCTION ---
                        # This returns a DataFrame with ['simulated_forecast', 'current_forecast', 'difference']
                        sim_result_df = generate_simulation(
                            level, selected_filter, exog_adjustments, sim_month, sim_key
                        )
                        
                        if not sim_result_df.empty:
                            month_num = {"October 2025": 10, "November 2025": 11, "December 2025": 12}.get(sim_month)
                            
                            if month_num is not None:
                                month_mask = merged_data['Month'] == month_num
                                if month_mask.any():
                                    # Extract values from the PKL result
                                    final_simulated = sim_result_df.iloc[0]['simulated_forecast']
                                    final_baseline = sim_result_df.iloc[0]['current_forecast_1']
                                    
                                    # Update Merged Data
                                    # 1. Update Simulated Column
                                    merged_data.loc[month_mask, 'simulated_forecast'] = final_simulated
                                    
                                    # 2. Update 'actual_forecast' (Baseline) to match PKL output exactly
                                    # This ensures 'Baseline' in chart = 'Model(Original Inputs)', not Excel
                                    merged_data.loc[month_mask, 'actual_forecast'] = final_baseline
                                    
                                    # Store for results card
                                    result_entry = {
                                        "month": sim_month,
                                        "selected_filter": selected_filter,
                                        "simulated": final_simulated,
                                        "baseline": final_baseline,
                                        "diff": final_simulated - final_baseline
                                    }
                                    simulation_results.append(result_entry)
                    
                    st.session_state[sim_key]['data'] = merged_data
                    st.session_state[sim_key]['generated'] = True
                    st.session_state[sim_key]['simulation_results'] = simulation_results
                    st.rerun()
                else:
                    st.info("ℹ️ No feature adjustments made. Please adjust at least one feature.")
                    st.session_state[sim_key]['generated'] = False
                    st.session_state.show_debug = False

        # Display simulation results
        if st.session_state[sim_key]['generated'] and st.session_state[sim_key]['data'] is not None:
            display_data = st.session_state[sim_key]['data']
            show_simulated = True
        else:
            display_data = base_forecast_data
            show_simulated = False

        # ==============================================================================
        # INSIGHT CARDS
        # ==============================================================================
        if sim_key in st.session_state and st.session_state[sim_key].get('simulation_results'):
            results = st.session_state[sim_key]['simulation_results']
            st.markdown("### 📊 Simulation Insight")
            
            cols = st.columns(len(results))
            for idx, row in enumerate(results):
                with cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"**{row['month']}**")
                        
                        baseline_val = row['baseline']
                        diff_val = row['diff']
                        
                        pct_change = (diff_val / baseline_val * 100) if baseline_val != 0 else 0
                        delta_str = f"{diff_val:,.0f} ({pct_change:+.2f}%)"
                        
                        st.metric(
                            label="Simulated Sales",
                            value=f"${row['simulated']:,.0f}",
                            delta=delta_str,
                            delta_color="normal"
                        )
                        st.caption(f"vs Baseline: {row['baseline']:,.0f}")

        # ==============================================================================
        # FORECAST VISUALIZATION (Chart & Table)
        # ==============================================================================
        st.markdown("---")
        st.subheader("📊 Forecast Visualization")
        
        control_col1, control_col2, control_col3, control_col4 = st.columns([2, 1, 1, 1])
        
        with control_col1:
            st.subheader("Forecast Chart")
            
        with control_col2:
            forecast_months = st.selectbox(
                "Forecast Period:",
                options=[1, 2, 3],
                index=2,
                format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
                key=f"{level}_{selected_filter}_forecast_period"
            )
            # Logic to handle period change...
            if f"prev_forecast_months_{level}_{selected_filter}" not in st.session_state:
                st.session_state[f"prev_forecast_months_{level}_{selected_filter}"] = forecast_months
                
            if st.session_state[f"prev_forecast_months_{level}_{selected_filter}"] != forecast_months:
                st.session_state[f"prev_forecast_months_{level}_{selected_filter}"] = forecast_months
                st.rerun()

        with control_col3:
            selected_years = st.multiselect(
                "Historical Years:",
                options=historical_years,
                default=historical_years,
                help="Select years to display in historical data",
                key=f"{level}_{selected_filter}_years"
            )

        with control_col4:
            view_mode = st.radio(
                "View Mode:",
                options=["Chart", "Table"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key=f"{level}_{selected_filter}_view_mode"
            )

        # Regenerate if months changed (standard logic)
        if forecast_months != len(base_forecast_data):
            with st.spinner(f"Regenerating forecast for {forecast_months} months..."):
                base_forecast_data = generate_future_forecast(
                    historical_data, level_filters, level, selected_filter, forecast_months=forecast_months
                )
                display_data = base_forecast_data
                st.session_state[sim_key]['generated'] = False
                st.session_state.show_debug = False
                st.session_state[sim_key]['simulation_results'] = None
                st.rerun()

        # Render Content
        if view_mode == "Chart":
            fig = create_forecast_plot(
                historical_data, 
                display_data, 
                selected_years, 
                show_simulated=show_simulated,
                level_filters=level_filters,
                level=level,
                selected_filter=selected_filter
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available to display.")
        else:
            if show_simulated and 'simulated_forecast' in display_data.columns:
                comparison_table = create_comparison_table(display_data)
                if comparison_table is not None:
                    st.dataframe(
                        comparison_table.style.format({
                            'Current Forecast': '{:,.2f}',
                            'Simulated Forecast': '{:,.2f}',
                            'Difference': '{:,.2f}',
                            'Change %': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
            else:
                if not display_data.empty:
                    display_table = display_data[['date', 'actual_forecast']].copy()
                    display_table['date'] = display_table['date'].dt.strftime('%Y-%m-%d')
                    display_table = display_table.rename(columns={
                        'date': 'Date',
                        'actual_forecast': 'Current Forecast'
                    })
                    st.dataframe(
                        display_table.style.format({
                            'Current Forecast': '{:,.2f}'
                        }),
                        use_container_width=True
                    )

        if st.session_state[sim_key]['generated'] and adjustments_made:
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.subheader("📋 Feature Adjustments Applied")
            feature_changes = []
            for feature_name, change in exog_adjustments.items():
                feature_display = get_business_name(feature_name)
                feature_changes.append({'Feature': feature_display, 'Month': selected_month, 'Change %': f"{change}%"})
            
            if feature_changes:
                changes_df = pd.DataFrame(feature_changes)
                st.dataframe(changes_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning(f"Please select at least one option for {level} level to continue.")
    
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """Load actual data from Databricks source"""
    try:
        with st.spinner("🔄 Loading data from Databricks..."):
            connection = sql.connect(
                server_hostname="adb-758907235096429.9.azuredatabricks.net",
                http_path="sql/protocolv1/o/758907235096429/1007-052243-r0241k4v",
                # access_token=st.secrets["DATABRICKS_TOKEN"]
                access_token = DATABRICKS_TOKEN
            )

            query = """
                SELECT Country, RBO, HVC, Date, Net_Sales
                FROM default.historical_sales_table
                WHERE Date <= '2025-09-30'
                ORDER BY Date, Country, RBO, HVC
            """
            df = pd.read_sql(query, connection)
            # st.dataframe(df)
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

            df = pd.DataFrame(result, columns=columns)

            if df.empty:
                return pd.DataFrame()

            df["date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["date"].dt.year
            df["Month"] = df["date"].dt.month
            df["Qtr"] = df["date"].dt.quarter
            df["MonthYear"] = df["date"].dt.strftime("%Y-%m")

            df = df.rename(columns={
                "Country": "country",
                "RBO": "rbo",
                "Net_Sales": "historical_sales"
            })

            return df

    except Exception as e:
        st.error("❌ Cannot access Databricks files. Please check configuration.")
        return pd.DataFrame()



# =============================================================================
# FORECAST PAGE
# =============================================================================
def forecast_page():
    embedded_param = get_query_param("embedded")
    if embedded_param:
        st.session_state.logged_in = True


    # Initialize file access check
    if not initialize_file_access():
        st.error("❌ Cannot proceed without Databricks file access")
        return


    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    
    historical_data = load_data()


    # Define filters
    country_filters = ["China", "Bangladesh", "Vietnam", "Honduras", "Turkey"]
    hvc_filters = ["Base", "Embelex", "Ipps", "Rfid"]
    rbo_filters = [
        "Adidas Grouped", "Decathlon", "Fast Retailing", "Gap Grouped", 
        "Hennes & Mauritz(H&M)", "Inditex Grouped", "Levi Grouped", 
        "Lululemon Athletica", "Marks & Spencer", "Nike Grouped"
    ]


    tab1, tab2, tab3, tab4 = st.tabs([" Country Level ", "HVC Level", "Top 50 RBO Level", "RBO Level"])


    with tab1:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 4])
        with col1:
            selected_countries = st.multiselect(
                "Select Countries:",
                options=country_filters,
                default=["China"],
                help="Choose countries for country-level forecasting",
                key="country_level_countries"
            )
            selected_filter = selected_countries[0] if selected_countries else None
            if selected_filter:
                st.markdown(f'<div class="success-message">✅ Successfully loaded country model for {selected_filter.upper()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="model-info-message">🔹 Country Level Model - Forecasting at country granularity</div>', unsafe_allow_html=True)
            level_filters = {'countries': selected_countries}
            create_level_forecast_tab(historical_data, "country", level_filters, selected_filter)
        st.markdown('</div>', unsafe_allow_html=True)


    with tab2:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader("HVC Level Forecasting")
        col1, col2 = st.columns([1, 4])
        with col1:
            selected_hvcs = st.multiselect(
                "Select HVCs:",
                options=hvc_filters,
                default=hvc_filters[:1] if hvc_filters else [],
                help="Choose HVCs for HVC-level forecasting",
                key="hvc_level_hvcs"
            )
            selected_filter = selected_hvcs[0] if selected_hvcs else None
            if selected_filter:
                st.markdown(f'<div class="success-message">✅ Successfully loaded HVC model for {selected_filter.upper()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="model-info-message">🔹 HVC Level Model - Forecasting at HVC granularity</div>', unsafe_allow_html=True)
            level_filters = {'hvcs': selected_hvcs}
            create_level_forecast_tab(historical_data, "hvc", level_filters, selected_filter)
        st.markdown('</div>', unsafe_allow_html=True)


    with tab3:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader("Top 50 RBO Level Forecasting")
        st.markdown(f'<div class="success-message">✅ Successfully loaded Top 50 RBO model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="model-info-message">🔹 Top 50 RBO Level Model - Forecasting for top 50 RBOs (No filters)</div>', unsafe_allow_html=True)
        
        level_filters = {}
        selected_filter = "top50_rbo"
        create_level_forecast_tab(historical_data, "top50_rbo", level_filters, selected_filter)
        st.markdown('</div>', unsafe_allow_html=True)


    with tab4:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.subheader("RBO Level Forecasting")
        col1, col2 = st.columns([1, 4])
        with col1:
            selected_rbos = st.multiselect(
                "Select RBOs:",
                options=rbo_filters,
                default=rbo_filters[:1] if rbo_filters else [],
                help="Choose RBOs for RBO-level forecasting",
                key="rbo_level_rbos"
            )
            selected_filter = selected_rbos[0] if selected_rbos else None
            if selected_filter:
                st.markdown(f'<div class="success-message">✅ Successfully loaded RBO model for {selected_filter.upper()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="model-info-message">🔹 RBO Level Model - Forecasting at RBO granularity</div>', unsafe_allow_html=True)
            level_filters = {'rbos': selected_rbos}
            create_level_forecast_tab(historical_data, "rbo", level_filters, selected_filter)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# OTHER PAGES
# =============================================================================
def home_page():
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.write("This app uses your actual Databricks model to generate future forecasts for 2025 based on historical data.")
    st.write("Go to the Forecast tab to start simulating with your real data.")
    st.markdown('</div>', unsafe_allow_html=True)


def login_page():
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    embedded_param = get_query_param("embedded")
    if embedded_param:
        st.session_state.logged_in = True
        st.rerun()
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                if username == "user" and password == "user":
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")


# =============================================================================
# NAVIGATION
# =============================================================================
def nav():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    embedded_param = get_query_param("embedded")
    if embedded_param:
        st.session_state.logged_in = True


    if logo_base64:
        header_html = f'''
        <div class="main-header">
          <div class="header-content">
            <div class="logo-container">
              <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="AD Logo">
            </div>
            <div class="title-container">
              <h1 class="header-title">Sales Forecast Simulator</h1>
            </div>
            <div style="width: 70px;"></div>
          </div>
        </div>
        '''
    else:
        header_html = '''
        <div class="main-header">
          <h1 class="header-title">Sales Forecast Simulator</h1>
        </div>
        '''
    
    st.markdown(header_html, unsafe_allow_html=True)


    if st.session_state.logged_in:
        page_options = ["Home", "Forecast"]
        selected_page = option_menu(
            None, page_options, icons=["house", "graph-up"],
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "white", "border-radius": "10px", "margin-bottom": "15px"},
                "icon": {"color": "#FF4500", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee", "color": "#FF4500"},
                "nav-link-selected": {"background-color": "#FF4500", "font-weight": "bold", "color": "white"},
            }
        )
        if selected_page == "Home":
            home_page()
        elif selected_page == "Forecast":
            forecast_page()
    else:
        login_page()


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # Initialize debug state
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = {}
    # Initialize max_features in session state from config
    if 'max_features' not in st.session_state:
        st.session_state['max_features'] = get_max_tweakable_features()
    
    nav()


if __name__ == "__main__":
    main()


