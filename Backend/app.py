import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom
from nilearn.image import resample_to_img
from monai.networks.nets import resnet10

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "multitask_brain_age.pth"
MNI_PATH = "mni_template.nii.gz"

AGE_MIN = 56.0
AGE_MAX = 91.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# MODEL
# =====================================================

class MultiTaskResNet(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2
        )

        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)
        )

        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        features = self.backbone(x)

        cls_logits = self.classifier(features)

        age_pred = self.regressor(features)

        return cls_logits, age_pred


# =====================================================
# LOAD MODEL
# =====================================================

model = MultiTaskResNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Model Loaded")

# =====================================================
# LOAD TEMPLATE
# =====================================================

mni = nib.load(MNI_PATH)

# =====================================================
# PREPROCESSING
# =====================================================

def normalize_percentile(img):

    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)

    img = np.clip(img, p1, p99)

    img = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-8
    )

    return img


def resize_volume(vol, target=(96,96,96)):

    factors = [
        target[0] / vol.shape[0],
        target[1] / vol.shape[1],
        target[2] / vol.shape[2]
    ]

    return zoom(
        vol,
        factors,
        order=1
    )


def preprocess_scan(path):

    img = nib.load(path)

    img = nib.as_closest_canonical(img)

    aligned = resample_to_img(
        source_img=img,
        target_img=mni,
        interpolation="continuous"
    )

    data = aligned.get_fdata()

    data = normalize_percentile(data)

    data = resize_volume(
        data,
        target=(96,96,96)
    )

    return data.astype(np.float32)

# =====================================================
# PREDICTION
# =====================================================

def predict_mri(file, chronological_age):

    volume = preprocess_scan(file)

    x = torch.tensor(
        volume,
        dtype=torch.float32
    )

    x = x.unsqueeze(0).unsqueeze(0)
    x = x.to(DEVICE)

    with torch.no_grad():

        cls_logits, age_pred = model(x)

        probs = F.softmax(
            cls_logits,
            dim=1
        )

        pred_class = torch.argmax(
            probs,
            dim=1
        ).item()

        cn_prob = probs[0][0].item()
        ad_prob = probs[0][1].item()

        age_norm = age_pred.item()

        brain_age = (
            age_norm
            *
            (AGE_MAX - AGE_MIN)
        ) + AGE_MIN

    diagnosis = (
        "CN"
        if pred_class == 0
        else "AD"
    )

    bag = brain_age - chronological_age

    result = f"""
Diagnosis: {diagnosis}

Brain Age: {brain_age:.2f} years

Chronological Age: {chronological_age:.2f} years

Brain Age Gap: {bag:.2f} years

CN Probability: {cn_prob:.4f}

AD Probability: {ad_prob:.4f}
"""

    return{"diagnosis": diagnosis,
                       "brain_age": brain_age,
                       "brain_age_gap": bag,
                       "cn_prob": cn_prob,
                       "ad_prob": ad_prob,
                       "result_text": result }

# =====================================================
# UI
# =====================================================

title = "Brain Age Gap Technology"

description = """
Upload a T1-weighted MRI (.nii or .nii.gz).

The model predicts:

• Brain Age  
• Brain Age Gap (BAG)  
• Alzheimer's Disease Classification  
• Prediction Confidence
"""
from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import shutil
import gradio as gr

# =====================================================
# GRADIO UI
# =====================================================
def gradio_predict(file, age):
    result = predict_mri(file, age)
    return result["result_text"]
demo = gr.Interface(
    fn=predict_mri,
    inputs=[
        gr.File(label="Upload MRI (.nii/.nii.gz)"),
        gr.Number(label="Chronological Age", value=70)
    ],
    outputs=gr.Textbox(label="Prediction"),
    title=title,
    description=description
)

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

@app.post("/predict")
async def predict_api(
        file: UploadFile = File(...),
        age: int = Form(...)
):
    try:

        suffix = (
            ".nii.gz"
            if file.filename.endswith(".nii.gz")
            else ".nii"
        )

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
        ) as tmp:

            shutil.copyfileobj(
                file.file,
                tmp
            )

            temp_path = tmp.name

        result = predict_mri(
            temp_path,
            age
        )

        return {
            "diagnosis": result["diagnosis"],
            "brain_age": result["brain_age"],
            "brain_age_gap": result["brain_age_gap"],
            "cn_prob": result["cn_prob"],
            "ad_prob": result["ad_prob"]
        }

    except Exception as e:

        return {
            "error": str(e)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
# Mount Gradio homepage
app = gr.mount_gradio_app(
    app,
    demo,
    path="/"
)