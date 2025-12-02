##Read Dataset
import pandas as pd, numpy as np, torch, torchvision as tv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
from PIL import Image
from warnings import warn
from Utility import HyperParameters

def get_df(csv):
    df = pd.read_csv(csv)
    cols = ["image_name","patient_id","sex","age_approx","anatom_site_general_challenge","target", 'patient_label']
    assert all(column in df.columns for column in cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
    df = df[cols].copy()
    return df

def group_split_df(df, test_size= 0.4, image_path = 'training_data/jpg'):
    gss = GroupShuffleSplit(n_splits=1, test_size= test_size, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["patient_id"]))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(df.iloc[temp_idx], groups=df.iloc[temp_idx]["patient_id"]))
    val_idx, test_idx  = temp_idx[val_idx], temp_idx[test_idx] # reindex to original df
    train_df, val_df, test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    train_df = pathify(train_df, image_path)
    val_df = pathify(val_df, image_path)
    test_df = pathify(test_df, image_path)
    return train_df, val_df, test_df

def pathify(frame, image_path):
    if image_path == 'training_data/jpg':
        return frame.assign(path=frame["image_name"].apply(lambda x: f"{image_path}/{x}.jpg"))
    elif image_path == 'training_data/pth':
        return frame.assign(path=frame["image_name"].apply(lambda x: f"{image_path}/{x}.pth"))
    else:
        raise ValueError("Only image_path/image_name and image_path/pth/image_name image or tensor file paths are supported. Ensure the images are in those locations. The image_path is set when PatientDS is instantiated. ")

# print("Train/Val/test size:", len(train_df), len(val_df), len(test_df))

def transforms(img_size):
    train_tfms = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(15),
        tv.transforms.GaussianBlur(3),
        tv.transforms.ToTensor(),
        tv.transforms.RandomErasing(),
        tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tfms = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_tfms = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_tfms, val_tfms, test_tfms



def encode_meta(frame, meta_cols, num_cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(frame[meta_cols])
    cat = enc.transform(frame[meta_cols])
    num = frame[num_cols].to_numpy(np.float32)
    num = (num - 0.0) / 100.0    # rough age scaling
    return np.concatenate([cat, num], axis=1).astype(np.float32)


# train_ds = MelanomaDS(train_df, train_tfms, meta=(train_meta if use_meta else None), mode='Train')
# val_ds   = MelanomaDS(val_df,   val_tfms,   meta=(val_meta   if use_meta else None), mode='Train')
# test_ds  = MelanomaDS(test_df,  test_tfms,   meta=(test_meta  if use_meta else None), mode = 'Test')

class PatientDS(Dataset, HyperParameters):
    def __init__(self, df, transform=None, image_path='training_data/jpg', use_meta=False, mode='train', saving_images=False):
        self.save_hyperparameters(ignore=['df','mode'])
        self.df = pathify(df, image_path)
        self.patient_ids = df["patient_id"].unique()
        self.groups = df.groupby("patient_id").indices
        self.mode = mode.lower()
        self.transform = transform
        if self.saving_images:
            warn(f"Saving images as tensors is enabled. Images will be iteratively saved to {image_path} when __getitem__ is called. Disable if this is not intended.")

        if self.transform is None:
            self.transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
            warn(f"No transform specified. Using default transform: {self.transform}",
                 UserWarning, stacklevel=2)

        if use_meta:
            meta_cols_cat = ["sex", "anatom_site_general_challenge"]
            meta_cols_num = ["age_approx"]

            for c in meta_cols_cat:
                self.df[c] = self.df[c].fillna("UNK")
            for c in meta_cols_num:
                self.df[c] = self.df[c].fillna(self.df[c].median())

            self.meta = encode_meta(self.df, meta_cols_cat, meta_cols_num)
        else:
            self.meta = None

    def __len__(self):
        return len(self.patient_ids)

    def get_image(self, row):
        if self.image_path == "training_data/jpg":
            img = Image.open(row.path).convert("RGB")
            x = self.transform(img)
            lesion_label = torch.tensor(row.target, dtype=torch.float32)
            patient_label = torch.tensor(row.patient_label, dtype=torch.float32)
            if self.meta is not None:
                m = torch.tensor(self.meta[row.Index], dtype=torch.float32)
            else:
                m = torch.empty(0, dtype=torch.float32)
            return x, patient_label, lesion_label, m

        elif self.image_path == "training_data/pth":
            x = torch.load(row.path)
        return x['img'],  x['patient_label'], x['lesion_label'], x['meta']


    def __getitem__(self, i):
        pid = self.patient_ids[i]
        rows = self.df.iloc[self.groups[pid]]

        imgs, patient_label, lesion_label_total, M = [], [], [], []

        for row in rows.itertuples(index=True):
            img, patient_label, lesion_label, m = self.get_image(row)
            imgs.append(img)
            lesion_label_total.append(lesion_label)
            M.append(m)
            if self.saving_images:
                warn('Saving images into "training_data/pth" as tensors. Stop if this is not intended.')
                sample = {
                    "img": img,
                    "meta": m.cpu(),
                    "lesion_label": lesion_label.cpu(),
                    'patient_label': patient_label.cpu()
                }
                torch.save(sample, f"training_data/pth/{row.image_name}.pth")


        lesion_labels = torch.stack(lesion_label_total)

        patient_label = torch.tensor(
            [1. if (lesion_labels == 1).any() else 0.],
            dtype=torch.float32
        )

        if self.meta is not None:
            M = torch.stack(M)
        else:
            M = torch.empty(0)

        return torch.stack(imgs), patient_label, lesion_labels, M


def get_sampler(df, oversample_ratio=3):
    # Reduce by patient_ID
    patient_df = df.groupby("patient_id")["patient_label"].max().reset_index()

    # Compute positive weight for BCE
    pos = (patient_df["patient_label"] == 1).mean()
    w_pos = (1 - pos) / max(pos, 1e-6)

    # Map to sample weights
    weights = patient_df["patient_label"].map({0: 1.0, 1: w_pos}).values
    weights = torch.tensor(weights, dtype=torch.float32)

    # Oversample positives by increasing num_samples
    num_samples = int(len(weights) * oversample_ratio)  # e.g., 3x dataset size
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    return sampler, w_pos

import torch
from torch.utils.data import Sampler
import numpy as np

class PatientLevelSampler(Sampler):
    def __init__(self, dataset, pos_fraction=0.5, num_samples=None):
        self.dataset = dataset
        self.pos_fraction = pos_fraction
        self.num_samples = num_samples or len(dataset)

        self.pos_indices = [i for i, pid in enumerate(dataset.patient_ids)
                            if dataset.df.loc[dataset.groups[pid], "patient_label"].max() == 1]
        self.neg_indices = [i for i in range(len(dataset)) if i not in self.pos_indices]

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("Dataset must contain both positive and negative patients!")

    def __iter__(self):
        for _ in range(self.num_samples):
            if np.random.rand() < self.pos_fraction:
                yield np.random.choice(self.pos_indices)
            else:
                yield np.random.choice(self.neg_indices)

    def __len__(self):
        return self.num_samples



# train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
# val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
# test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)


