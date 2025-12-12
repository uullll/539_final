import sys
import os
from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. Environment setup and imports
# ==========================================
# Ensure project root is on sys.path so `src` can be imported
proj_root = Path().resolve()
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

# Import model definitions from `src`
# Note: ensure `src/models.py` exists and defines these classes
try:
    from src.models import ResNet101Meta as ResNetModel, Efficientnet_b6
except ImportError as e:
    st.error(f"Failed to import model code: {e}")
    st.stop()

# Default metadata dimension used by the models (tunable based on how model was trained)
METADATA_DIM = 3

# ==========================================
# 2. tool functions for checkpoint loading
# ==========================================
def _load_checkpoint(path: str):
    """Safely load a checkpoint file; support several save formats."""
    try:
        # try loading with weights_only first (PyTorch 2.0+)
        try:
            ck = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            ck = torch.load(path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}")

    # deal with different checkpoint formats
    if isinstance(ck, dict) and 'state_dict' in ck and isinstance(ck['state_dict'], dict):
        sd = ck['state_dict']
    elif isinstance(ck, dict):
        sd = ck
    else:
        try:
            sd = ck.state_dict()
        except Exception:
            raise RuntimeError('Unrecognized checkpoint format')
    return sd

def _normalize_key(k: str) -> str:
    """Remove a leading 'module.' prefix if present."""
    if k.startswith('module.'):
        k = k[len('module.'):]
    return k

def _map_keys_to_model(ck_dict: dict, model_keys: set) -> dict:
    """Try to map checkpoint keys to the model's state_dict keys using common heuristics."""
    prefixes = ['resnet101.', 'resnet.', 'backbone.', 'efficientnet_b6.', 'efficientnet.', 'model.', 'net.', 'encoder.']
    new = {}
    for k, v in ck_dict.items():
        k2 = _normalize_key(k)
        if k2 in model_keys:
            new[k2] = v
            continue
        found = False
        for p in prefixes:
            if k2.startswith(p):
                kk = k2[len(p):]
                if kk in model_keys:
                    new[kk] = v
                    found = True
                    break
        if found:
            continue
        # finally try endswith matching
        for mk in model_keys:
            if mk.endswith(k2) or k2.endswith(mk):
                new[mk] = v
                found = True
                break
    return new

# ==========================================
# 3. Core loading logic (includes metadata-dimension handling)
# ==========================================
@st.cache_resource
def load_model():
    # Candidate checkpoint files (place your .pt/.pth files in one of these paths)
    candidates = [
        ('model/Best_Efficient_net_B6_2.pt', 'efficientnet'),
        ('model/Best_Efficient_net_B6.pt', 'efficientnet'),
        ('model/saved_model.pth', 'efficientnet'),  # assume saved_model is EfficientNet
        ('model/Best_Resnet101_meta.pt', 'resnet101'),
        # also check at repository root
        ('Best_Efficient_net_B6.pt', 'efficientnet'),
        ('saved_model.pth', 'efficientnet'),
    ]

    ck_path = None
    ck_type = None
    for fn, tp in candidates:
        if Path(fn).exists():
            ck_path = fn
            ck_type = tp
            break

    if ck_path is None:
        st.error("No model file found! Please place a .pt or .pth checkpoint in the project directory.")
        raise FileNotFoundError('No checkpoint found.')

    print(f'Found checkpoint: {ck_path} (Type: {ck_type})')

    # === Key fix: set metadata dimension used by the model ===
    # If your checkpoint was trained with extra patient/meta features, set METADATA_DIM accordingly.
    # Historically, a mismatch of 2320 vs 2304 implied 3 metadata features; default METADATA_DIM=3 is used.

    # Instantiate the appropriate model
    if 'efficientnet' in ck_type:
        model = Efficientnet_b6(meta_dim=METADATA_DIM)
    else:
        model = ResNetModel(meta_dim=METADATA_DIM)

    # Load and normalize checkpoint keys
    sd = _load_checkpoint(ck_path)
    sd_norm = { _normalize_key(k): v for k, v in sd.items() }

    model_keys = set(model.state_dict().keys())
    mapped = _map_keys_to_model(sd_norm, model_keys)

    # Try loading mapped keys (non-strict to be tolerant of extras/missing)
    try:
        model.load_state_dict(mapped, strict=False)
    except Exception as e:
        print('Mapped load failed, trying raw normalized dict with strict=False:', e)
        model.load_state_dict(sd_norm, strict=False)

    model.eval()
    return model, ck_type

# ==========================================
# 4. frontend (Streamlit app)
# ==========================================
st.set_page_config(page_title="Melanoma AI Detector", page_icon="🩺")

st.title("🩺 Melanoma AI Detector")
st.markdown("""
This web app uses Deep Learning models (EfficientNet / ResNet) to assist with dermoscopy image analysis.
Upload an image to get a malignancy risk estimate.
""")

# Load model
try:
    with st.spinner('Loading AI model (may take a few seconds)...'):
        model, model_type = load_model()
    st.success(f"Model loaded successfully! (Architecture: {model_type})")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# upload image
uploaded_file = st.file_uploader(" Upload skin image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # show uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded image', use_column_width=True)
    
    with col2:
        st.write("### Ready to analyze")
        if st.button('Start AI analysis', type="primary"):
            with st.spinner('AI computing features...'):
                # 1. Image preprocessing
                img_tensor = preprocess(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

                # 2. Construct dummy metadata (use METADATA_DIM used during model instantiation)
                meta_data = torch.zeros((1, METADATA_DIM))

                # 3. Inference
                with torch.no_grad():
                    try:
                        # Try calling model(image, meta)
                        output = model(img_tensor, meta_data)
                    except TypeError:
                        # If model.forward only accepts image
                        output = model(img_tensor)
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    # 4. Post-process output: decide sigmoid vs softmax
                    if output.shape[1] == 1:
                        # Binary output (BCEWithLogits)
                        probs = torch.sigmoid(output)
                        malignant_prob = probs.item()
                    else:
                        # Multi-class output (CrossEntropy)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        # Assume index 1 corresponds to Malignant (adjust per your class mapping)
                        malignant_prob = probs[0][1].item()
            
            # 5. Display results
            st.divider()
            st.metric(label="Malignancy Probability", value=f"{malignant_prob:.2%}")

            st.progress(malignant_prob)

            if malignant_prob > 0.5:
                st.error("shit **High Risk**: Recommend immediate consultation with a dermatologist.")
            else:
                st.success("nice **Low Risk**: No obvious malignant features detected.")