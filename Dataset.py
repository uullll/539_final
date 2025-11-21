##Read Dataset
import pandas as pd, numpy as np, torch, torchvision as tv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

##Configuration setting##
# COMP_DIR =""
# IMG_SIZE=224
# batch=100
# epochs1=1
# epochs2=2
# use_meta=True
# num_workers=0 ##
# device   = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Device:", device)


def get_df(csv):
    df = pd.read_csv(csv)
    cols = ["image_name","patient_id","sex","age_approx","anatom_site_general_challenge","target"]
    assert all(column in df.columns for column in cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
    df = df[cols].copy()
    return df

def group_split_df(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["patient_id"]))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(df.iloc[temp_idx], groups=df.iloc[temp_idx]["patient_id"]))
    val_idx, test_idx  = temp_idx[val_idx], temp_idx[test_idx] # reindex to original df
    train_df, val_df, test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    train_df = pathify(train_df, "train")
    val_df = pathify(val_df, "train")
    test_df = pathify(test_df, "train")
    return train_df, val_df, test_df

def pathify(frame, split):
    return frame.assign(path=frame["image_name"].apply(lambda x: f"jpeg/{split}/{x}.jpg"))

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



def encode_meta(frame, enc, meta_cols, num_cols):
    cat = enc.transform(frame[meta_cols])
    num = frame[num_cols].to_numpy(np.float32)
    num = (num - 0.0) / 100.0    # rough age scaling
    return np.concatenate([cat, num], axis=1).astype(np.float32)


class MelanomaDS(Dataset):
    def __init__(self, frame, tfm, use_meta=False, mode='Train'):
        super().__init__()
        self.df, self.tfm, self.use_meta, self.mode = frame.reset_index(drop=True), tfm, use_meta, mode
        if self.use_meta:
            meta_cols = ["sex", "anatom_site_general_challenge"]
            num_cols = ["age_approx"]
            # Fill categorical missing values
            for c in meta_cols:
                self.df[c] = self.df[c].fillna("UNK")

            # Fill numeric missing values with median
            for c in num_cols:
                med = self.df[c].median()
                self.df[c] = self.df[c].fillna(med)

            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(self.df[meta_cols])
            self.meta = encode_meta(self.df, enc, meta_cols, num_cols)

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.loc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tfm(img)
        if self.mode == "Train":
            y = torch.tensor(row["target"], dtype=torch.float32)
            m = torch.tensor(self.meta[i], dtype=torch.float32) if (self.use_meta and self.meta is not None) else torch.tensor([])
            return x, y, m
        elif self.mode == 'Test':
            y = torch.tensor(row["target"], dtype=torch.float32)
            m = torch.tensor(self.meta[i], dtype=torch.float32) if (self.use_meta and self.meta is not None) else torch.tensor([])
            return x, y, m, i
        else:
            m = torch.tensor(self.meta[i], dtype=torch.float32) if (self.use_meta and self.meta is not None) else torch.tensor([])
            return x, row["image_name"], m

# train_ds = MelanomaDS(train_df, train_tfms, meta=(train_meta if use_meta else None), mode='Train')
# val_ds   = MelanomaDS(val_df,   val_tfms,   meta=(val_meta   if use_meta else None), mode='Train')
# test_ds  = MelanomaDS(test_df,  test_tfms,   meta=(test_meta  if use_meta else None), mode = 'Test')

def sampler(train_df):
    pos = (train_df["target"]==1).mean()
    w_pos = (1 - pos) / max(pos, 1e-6) # Define weights for data based on relative abundance of positives
    weights = train_df["target"].map({0:1.0, 1:w_pos}).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, w_pos

# train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
# val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
# test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)