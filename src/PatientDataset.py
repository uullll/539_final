##Read Dataset
import pandas as pd, numpy as np, torch, torchvision as tv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset
from PIL import Image
from warnings import warn
from Utility import HyperParameters

def get_df(csv = None):
    if csv is not None:
        df = pd.read_csv(csv)
        cols = ["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge", "target"]
        assert all(column in df.columns for column in
                   cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
        df = df[cols].copy()
    else:
        ds = MelanomaDS_19_20()
        df = ds.df_2020
        df.drop(columns=['BCC', 'AK', 'SCC', 'VASC', 'DF'], axis=1,
                      inplace=True)  # drop columns not present in the 2020 dataset
    return df

def group_split_df(df, test_size= 0.2, image_path = 'train/2020_jpg'):
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
    if image_path.endswith('jpg'):
        return frame.assign(path=frame["image"].apply(lambda x: f"{image_path}/{x}.jpg"))
    elif image_path.endswith('pth'):
        return frame.assign(path=frame["image"].apply(lambda x: f"{image_path}/{x}.pth"))
    else:
        raise ValueError("Check file path. image_path is set when PatientDS is instantiated.")

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

import warnings
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tv
import pandas as pd


class PatientDS(Dataset, HyperParameters):
    def __init__(self, transform=None, image_path = None, csv_path = 'ISIC_2020_Training_GroundTruth_v2.csv',
                 use_meta=False, mode= 'full', saving_images=False, upsampling=False, diagnosis = True):
        self.save_hyperparameters()
        self.mode = mode.lower()
        self.df = self.get_df2020(csv_path)
        self.df = self.df.rename(columns={"anatom_site_general_challenge": "anatom_site_general", "image_name": "image" })
        self.df = pathify(self.df, image_path)
        self.label_cols = ["MEL", "NV", "BKL", "BMP"]

        # split groups
        self.train_df, self.val_df, self.test_df = group_split_df(self.df, test_size=0.2, image_path=image_path)

        self.splits = {'train':self.train_df, 'val':self.val_df, 'test':self.test_df, 'full':self.df  }
        self._get_split() # select split according to 'mode'

        if self.saving_images:
            warnings.warn(
                f"Saving images as tensors is enabled. Images will be iteratively saved to {image_path} when __getitem__ is called."
            )
        if self.mode != 'train' and self.upsampling is True:
            self.upsampling = False
            warnings.warn("No upsampling because mode is not 'train'")

        if self.transform is None:
            if self.upsampling:
                warnings.warn("No upsampling because transformation is None.")
                self.upsampling = False
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            warnings.warn(f"No transform specified. Using default transform: {self.transform}", UserWarning, stacklevel=2)

        if use_meta:
            meta_cols_cat = ["sex", "anatom_site_general"]
            meta_cols_num = ["age_approx"]

            for c in meta_cols_cat:
                self.df[c] = self.df[c].fillna("UNK")
            for c in meta_cols_num:
                self.df[c] = self.df[c].fillna(self.df[c].median())

            self.meta = encode_meta(self.df, meta_cols_cat, meta_cols_num)
        else:
            self.meta = None

    def _get_split(self):
        self.df = self.splits[self.mode]
        self._process_split()

    def _process_split(self):
        self.sum_pos_lesions = self.df.groupby('patient_id')['target'].sum()
        self.groups = self.df.groupby("patient_id").indices
        self.df_patient = (
            self.df.groupby("patient_id", as_index=False)["target"].max())  # collapse dataframe by patient_id
        self.patient_ids = self.df_patient["patient_id"].values
        

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, i):
        pid = self.patient_ids[i]
        rows = self.df.iloc[self.groups[pid]]

        # Upsample positives if requested
        if self.upsampling:
            rows = balance_patient(rows)

        imgs, lesion_label_total, M = [], [], []

        if self.diagnosis: # get multiclass target from diagnosis column
            lesion_labels = rows[self.label_cols].to_numpy(dtype=np.float32)
            lesion_labels = np.max(lesion_labels, axis=0) # if any diagnosis class is 1 for any lesion, assign 1 to that patient's diagnosis class for all the patient's diagnosis classes (here = 4)
            lesion_labels = torch.from_numpy(lesion_labels)
            for row in rows.itertuples(index=True):
                img, _, m = self._get_image(row)
                imgs.append(img)
                M.append(m)
            if self.meta is not None:
                M = torch.stack(M)
            else:
                M = torch.empty((0,), dtype=torch.float32)

            return torch.stack(imgs), lesion_labels, M


        else: # Get target only from melanoma class
            for row in rows.itertuples(index=True):
                img, lesion_label, m = self._get_image(row)
                imgs.append(img)
                lesion_label_total.append(lesion_label)
                M.append(m)

                if self.saving_images:
                    warnings.warn('Saving images into "training_data/2020/pth" as tensors. Stop if this is not intended.')
                    sample = {
                        "img": img,
                        "meta": m.cpu(),
                        "lesion_label": lesion_label.cpu()
                    }
                    torch.save(sample, f"training_data/2020/pth/{row.Index}.pth")
            lesion_labels = torch.stack(lesion_label_total)
            if self.meta is not None:
                M = torch.stack(M)
            else:
                M = torch.empty((0,), dtype=torch.float32)

            patient_label = torch.tensor([rows['target'].iloc[0]], dtype=torch.float32)

            return torch.stack(imgs), patient_label, M # lesion_labels,--> could also export

    def _get_image(self, row):
        if row.path.endswith(".jpg"):
            img = Image.open(row.path).convert("RGB")
            x = self.transform(img)
            lesion_label = torch.tensor(row.target, dtype=torch.float32)

            if self.meta is not None:
                m = torch.tensor(self.meta[row.Index], dtype=torch.float32)
            else:
                m = torch.empty(0, dtype=torch.float32)

            return x, lesion_label, m

        elif row.path.endswith(".pth"):
            x = torch.load(row.path)
            return x['img'], x['lesion_label'], x['meta']
        else:
            raise FileNotFoundError(
                'Data folder not found.'
            )

    def get_df2020(self, path):
        self.bkl = {'seborrheic keratosis', 'lentigo NOS', 'lichenoid keratosis', 'solar lentigo'} # benign keratosis or lentigo
        self.BMP = {'cafe-au-lait macule', 'atypical melanocytic proliferation'} # Benign macule or proliferation
        df = self._get_df2020(path)

        df["diagnosis_collapsed"] = df["diagnosis"].apply(self.collapse_col)
        classes = ["MEL", "NV", "BKL", "BMP"]

        # create binary columns
        for c in classes:
            df[c] = (df["diagnosis_collapsed"] == c).astype(int)

        # drop original labels
        df = df.drop(columns=["diagnosis", "diagnosis_collapsed"])
        return df

    def collapse_col(self, x):
        if x in self.bkl:
            return "BKL"
        if x in self.BMP:
            return "BMP"
        if x == 'nevus':
            return "NV"
        if x == 'melanoma':
            return "MEL"
        # if x == 'unknown':
        #     return "UNK"
        return x

    def _get_df2020(self, csv):
        df = pd.read_csv(csv)
        cols = ["image_name", "patient_id", "lesion_id", "sex", "age_approx", "anatom_site_general_challenge",
                "diagnosis", "benign_malignant", "target"]
        assert all(column in df.columns for column in
                   cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
        df = df[cols].copy()
        return df

def balance_patient(group):
    pos = group[group['target'] == 1]
    neg = group[group['target'] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return group

    if len(pos) < len(neg):
        pos_upsampled = pos.sample(len(neg), replace=True)
        return pd.concat([neg, pos_upsampled])
    else:
        return group





def unnormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # undo normalize
    t = t * std + mean
    # clamp to valid range
    t = t.clamp(0, 1)
    # convert to PIL
    return TF.to_pil_image(t)



