##Read Dataset
import pandas as pd, numpy as np, torch, torchvision as tv
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from PIL import Image
from warnings import warn
from Utility import HyperParameters
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


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
    return frame.assign(path=frame["image_name"].apply(lambda x: f"train/{split}/{x}.jpg"))

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

def make_splits(df19, df20):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train20_idx, temp20_idx = next(gss.split(df20, groups=df20["patient_id"]))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val20_idx, test20_idx = next(gss2.split(df20.iloc[temp20_idx],
                                            groups=df20.iloc[temp20_idx]["patient_id"]))

    val20_idx  = temp20_idx[val20_idx]
    test20_idx = temp20_idx[test20_idx]

    train20 = df20.iloc[train20_idx].reset_index(drop=True)
    val20   = df20.iloc[val20_idx].reset_index(drop=True)
    test20  = df20.iloc[test20_idx].reset_index(drop=True)

    train19, temp19 = train_test_split(df19,
                                       test_size=0.2,
                                       random_state=42)

    val19, test19 = train_test_split(temp19,
                                    test_size=0.5,
                                    random_state=42)

    train19 = train19.reset_index(drop=True)
    val19   = val19.reset_index(drop=True)
    test19  = test19.reset_index(drop=True)


    return (train19, val19, test19,
        train20, val20, test20)


class cross_MelanomaDS_19_20(Dataset):

    def __init__(self, use_meta=True):
        super().__init__()

        self.use_meta = use_meta

        self.df_2019, self.datacols, self.metacols = self.get_df2019()
        self.df_2019 = self.pathify(self.df_2019, "image", "2019_jpg")
        self.df_2020 = self.get_df2020()
        self.df_2020 = self.df_2020.rename(
            columns={"anatom_site_general_challenge": "anatom_site_general", "image_name": "image", }
        )
        self.df_2020 = self.pathify(self.df_2020, "image", "2020_jpg")

        # Remove duplicates from 2020 data
        df_duplicates = pd.read_csv('ISIC_2020_Training_Duplicates.csv')
        images_to_remove = df_duplicates['image_name_2'].unique()
        self.df_2020 = self.df_2020[~self.df_2020['image'].isin(images_to_remove)].copy()
        self.df_2020.reset_index(drop=True, inplace=True)

        # Metadata
        self.meta_cols = ["anatom_site_general", "sex"]
        self.num_cols = ["age_approx"]

        # Data
        self.label_cols = ["MEL", "NV", "BKL", "DF",
                           "VASC", "SCC", "AK", "BCC", "UNK"]

        self.df_2019_meta = self.get_meta(self.df_2019)
        self.df_2020_meta = self.get_meta(self.df_2020)

        self.df = None
        self.meta = None
        self.transform = None

    @classmethod
    def full(cls, transform=None):
        obj = cls()
        if transform is not None:
            obj.transform = transform

        obj.df = pd.concat([obj.df_2019, obj.df_2020], ignore_index=True)

        meta = pd.concat([obj.df_2019_meta,
                          obj.df_2020_meta],
                         ignore_index=True)

        obj.meta = obj.encode_meta(meta, obj.meta_cols, obj.num_cols)
        return obj

    @classmethod
    def train(cls, transform):
        return cls._make_split(transform, "train")

    @classmethod
    def val(cls, transform):
        return cls._make_split(transform, "val")

    @classmethod
    def test(cls, transform):
        return cls._make_split(transform, "test")

    @classmethod
    def _make_split(cls, transform, split):
        obj = cls()
        obj.transform = transform

        (t19, v19, te19,
         t20, v20, te20) = make_splits(obj.df_2019, obj.df_2020)

        split_map = {"train": (t19, t20), "val": (v19, v20), "test": (te19, te20)}
        df19, df20 = split_map[split]

        df = pd.concat([df19, df20], ignore_index=True)
        meta = pd.concat([obj.df_2019_meta.loc[df19.index], obj.df_2020_meta.loc[df20.index], ], ignore_index=True)

        meta = obj.encode_meta(meta, obj.meta_cols, obj.num_cols)

        obj.df = df
        obj.meta = meta
        return obj

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]

        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)

        labels = row[self.label_cols].astype(float).to_numpy()
        y = torch.tensor(labels, dtype=torch.float32)


        if self.use_meta and self.meta is not None:
            m = torch.tensor(self.meta[i], dtype=torch.float32)
        else:
            m = torch.tensor([])

        return x, y, m

    def pathify(self, frame, image_col, split):
        return frame.assign(path="train/" + split + "/" + frame[image_col] + ".jpg")

    def encode_meta(self, frame, meta_cols, num_cols):

        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1)

        enc.fit(frame[meta_cols])
        cat = enc.transform(frame[meta_cols])

        num = frame[num_cols].to_numpy(np.float32)
        num = num / 100.0

        return np.concatenate([cat, num], axis=1).astype(np.float32)

    def get_df2019(self, csv_data="ISIC_2019_Training_GroundTruth.csv", csv_meta="ISIC_2019_Training_Metadata.csv"):
        assert csv_data.endswith("ISIC_2019_Training_GroundTruth.csv")
        assert csv_meta.endswith("ISIC_2019_Training_Metadata.csv")
        df_data = pd.read_csv(csv_data)
        df_meta = pd.read_csv(csv_meta)
        df = pd.merge(df_data, df_meta, on='image', how='left')
        cols = [
            "image", "sex", "age_approx", "anatom_site_general",
            "lesion_id", "MEL", "NV", "BCC", "AK", "BKL",
            "DF", "VASC", "SCC", "UNK"]
        missing = [c for c in cols if c not in df.columns]
        assert len(missing) == 0, f"Missing columns: {missing}\nCurrent columns: {df.columns}"
        df = df[cols].copy()
        return df, df_data.columns.tolist(), df_meta.columns.tolist()

    def get_df2020(self, path='ISIC_2020_Training_GroundTruth_v2.csv'):
        self.bkl = {'seborrheic keratosis', 'lentigo NOS', 'lichenoid keratosis', 'solar lentigo'}
        self.unknown = {'cafe-au-lait macule', 'atypical melanocytic proliferation'}

        self.bkl = {s.lower() for s in self.bkl}
        self.unknown = {s.lower() for s in self.unknown}

        df = self._get_df2020(path)

        df["diagnosis_collapsed"] = df["diagnosis"].apply(self.collapse_col)
        classes = ["MEL", "NV", "BKL", "DF", "VASC", "SCC", "AK", "BCC", "UNK"]

        # create binary columns
        for c in classes:
            df[c] = (df["diagnosis_collapsed"] == c).astype(int)

        # drop original labels
        df = df.drop(columns=["diagnosis", "diagnosis_collapsed"])
        return df

    def collapse_col(self, x):
        if x.lower() in self.bkl:
            return "BKL"
        if x.lower() in self.unknown:
            return "UNK"
        if x.lower() == "melanoma":
            return "MEL"
        if x.lower() == "nevus":
            return "NV"
        if x.lower() == "dermatofibroma":
            return "DF"
        if x.lower() == "vascular lesion":
            return "VASC"
        if x.lower() == "squamous cell carcinoma":
            return "SCC"
        if x.lower() == "actinic keratosis":
            return "AK"
        if x.lower() == "basal cell carcinoma":
            return "BCC"
        return "UNK"

    def _get_df2020(self, csv):
        df = pd.read_csv(csv)
        cols = ["image_name", "patient_id", "lesion_id", "sex", "age_approx", "anatom_site_general_challenge",
                "diagnosis", "benign_malignant", "target"]
        assert all(column in df.columns for column in
                   cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
        df = df[cols].copy()
        return df

    def get_meta(self, df):
        if self.use_meta:
            df = df.copy()
            # Fill categorical missing values
            for c in self.meta_cols:
                df[c] = df[c].fillna("UNK")

            # Fill numeric missing values with median
            for c in self.num_cols:
                med = df[c].median()
                df[c] = df[c].fillna(med)
            return df
        else:
            return df

    def example_images(self, i):
        row = self.df.loc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)

        labels = row[self.label_cols].astype(float).to_numpy()
        y = torch.tensor(labels, dtype=torch.float32)

        if self.use_meta and self.meta is not None:
            m = torch.tensor(self.meta[i], dtype=torch.float32)
        else:
            m = torch.tensor([])
        # Convert transformed tensor back to image
        # x_pil = TF.to_pil_image(x)
        x_unnorm = unnormalize(x)  #

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Unnormalized Image")
        plt.imshow(x_unnorm)
        plt.axis("off")

        plt.show()
        return y, m

class multiclass_BCE_MelanomaDS_19_20(Dataset):
    def __init__(self, use_meta=True):
        super().__init__()

        self.use_meta = use_meta

        self.df_2019, self.datacols, self.metacols = self.get_df2019()
        self.df_2019 = self.pathify(self.df_2019, "image", "2019_jpg")
        self.df_2020 = self.get_df2020()
        self.df_2020 = self.df_2020.rename(
            columns={"anatom_site_general_challenge": "anatom_site_general", "image_name": "image", }
        )
        self.df_2020 = self.pathify(self.df_2020, "image", "2020_jpg")
        # Metadata
        self.meta_cols = ["anatom_site_general", "sex"]
        self.num_cols = ["age_approx"]

        # Data
        self.label_cols = ["MEL", "NV", "BKL", "DF",
                           "VASC", "SCC", "AK", "BCC", "UNK"]

        self.df_2019_meta = self.get_meta(self.df_2019)
        self.df_2020_meta = self.get_meta(self.df_2020)

        self.df = None
        self.meta = None
        self.transform = None

    @classmethod
    def full(cls, transform=None):
        obj = cls()
        if transform is not None:
            obj.transform = transform

        obj.df = pd.concat([obj.df_2019, obj.df_2020], ignore_index=True)

        meta = pd.concat([obj.df_2019_meta,
                          obj.df_2020_meta],
                         ignore_index=True)

        obj.meta = obj.encode_meta(meta, obj.meta_cols, obj.num_cols)
        return obj

    @classmethod
    def train(cls, transform):
        return cls._make_split(transform, "train")

    @classmethod
    def val(cls, transform):
        return cls._make_split(transform, "val")

    @classmethod
    def test(cls, transform):
        return cls._make_split(transform, "test")

    @classmethod
    def _make_split(cls, transform, split):
        obj = cls()
        obj.transform = transform

        (t19, v19, te19,
         t20, v20, te20) = make_splits(obj.df_2019, obj.df_2020)

        split_map = {"train": (t19, t20), "val": (v19, v20), "test": (te19, te20)}
        df19, df20 = split_map[split]

        df = pd.concat([df19, df20], ignore_index=True)
        meta = pd.concat([obj.df_2019_meta.loc[df19.index], obj.df_2020_meta.loc[df20.index], ], ignore_index=True)

        meta = obj.encode_meta(meta, obj.meta_cols, obj.num_cols)

        obj.df = df
        obj.meta = meta
        return obj

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]

        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)

        labels = row[self.label_cols].astype(float).to_numpy()
        y = torch.tensor(labels, dtype=torch.float32)


        if self.use_meta and self.meta is not None:
            m = torch.tensor(self.meta[i], dtype=torch.float32)
        else:
            m = torch.tensor([])

        return x, y, m

    def pathify(self, frame, image_col, split):
        return frame.assign(path="train/" + split + "/" + frame[image_col] + ".jpg")

    def encode_meta(self, frame, meta_cols, num_cols):

        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1)

        enc.fit(frame[meta_cols])
        cat = enc.transform(frame[meta_cols])

        num = frame[num_cols].to_numpy(np.float32)
        num = num / 100.0

        return np.concatenate([cat, num], axis=1).astype(np.float32)

    def get_df2019(self, csv_data="ISIC_2019_Training_GroundTruth.csv", csv_meta="ISIC_2019_Training_Metadata.csv"):
        assert csv_data.endswith("ISIC_2019_Training_GroundTruth.csv")
        assert csv_meta.endswith("ISIC_2019_Training_Metadata.csv")
        df_data = pd.read_csv(csv_data)
        df_meta = pd.read_csv(csv_meta)
        df = pd.merge(df_data, df_meta, on='image', how='left')
        cols = [
            "image", "sex", "age_approx", "anatom_site_general",
            "lesion_id", "MEL", "NV", "BCC", "AK", "BKL",
            "DF", "VASC", "SCC", "UNK"]
        missing = [c for c in cols if c not in df.columns]
        assert len(missing) == 0, f"Missing columns: {missing}\nCurrent columns: {df.columns}"
        df = df[cols].copy()
        return df, df_data.columns.tolist(), df_meta.columns.tolist()

    def get_df2020(self, path='ISIC_2020_Training_GroundTruth_v2.csv'):
        self.bkl = {'seborrheic keratosis', 'lentigo NOS', 'lichenoid keratosis', 'solar lentigo'}
        self.unknown = {'cafe-au-lait macule', 'atypical melanocytic proliferation'}
        df = self._get_df2020(path)

        df["diagnosis_collapsed"] = df["diagnosis"].apply(self.collapse_col)
        classes = ["MEL", "NV", "BKL", "DF", "VASC", "SCC", "AK", "BCC", "UNK"]

        # create binary columns
        for c in classes:
            df[c] = (df["diagnosis_collapsed"] == c).astype(int)

        # drop original labels
        df = df.drop(columns=["diagnosis", "diagnosis_collapsed"])
        return df

    def collapse_col(self, x):
        if x in self.bkl:
            return "BKL"
        if x in self.unknown:
            return "UNK"
        if x.lower() == "melanoma":
            return "MEL"
        if x.lower() == 'nevus':
            return "NV"
        return x

    def _get_df2020(self, csv):
        df = pd.read_csv(csv)
        cols = ["image_name", "patient_id", "lesion_id", "sex", "age_approx", "anatom_site_general_challenge",
                "diagnosis", "benign_malignant", "target"]
        assert all(column in df.columns for column in
                   cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
        df = df[cols].copy()
        return df

    def get_meta(self, df):
        if self.use_meta:
            df = df.copy()
            # Fill categorical missing values
            for c in self.meta_cols:
                df[c] = df[c].fillna("UNK")

            # Fill numeric missing values with median
            for c in self.num_cols:
                med = df[c].median()
                df[c] = df[c].fillna(med)
            return df
        else:
            return df

    def example_images(self, i):
        row = self.df.loc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)

        labels = row[self.label_cols].astype(float).to_numpy()
        y = torch.tensor(labels, dtype=torch.float32)

        if self.use_meta and self.meta is not None:
            m = torch.tensor(self.meta[i], dtype=torch.float32)
        else:
            m = torch.tensor([])
        # Convert transformed tensor back to image
        # x_pil = TF.to_pil_image(x)
        x_unnorm = unnormalize(x)  #

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Unnormalized Image")
        plt.imshow(x_unnorm)
        plt.axis("off")

        plt.show()
        return y, m

class BCE_MelanomaDS_19_20(Dataset):
    def __init__(self, use_meta=True):
        super().__init__()

        self.use_meta = use_meta

        self.df_2019, self.datacols, self.metacols = self.get_df2019()
        self.df_2019 = self.pathify(self.df_2019, "image", "2019_jpg")
        self.df_2020 = self.get_df2020()
        self.df_2020 = self.df_2020.rename(
            columns={"anatom_site_general_challenge": "anatom_site_general", "image_name": "image", }
        )
        self.df_2020 = self.pathify(self.df_2020, "image", "2020_jpg")
        # Metadata
        self.meta_cols = ["anatom_site_general", "sex"]
        self.num_cols = ["age_approx"]

        # Data
        self.label_cols = ["MEL", "NV", "BKL", "DF",
                           "VASC", "SCC", "AK", "BCC", "UNK"]

        self.df_2019_meta = self.get_meta(self.df_2019)
        self.df_2020_meta = self.get_meta(self.df_2020)

        self.df = None
        self.meta = None
        self.transform = None

    @classmethod
    def full(cls, transform=None):
        obj = cls()
        if transform is not None:
            obj.transform = transform

        obj.df = pd.concat([obj.df_2019, obj.df_2020], ignore_index=True)

        meta = pd.concat([obj.df_2019_meta,
                          obj.df_2020_meta],
                         ignore_index=True)

        obj.meta = obj.encode_meta(meta, obj.meta_cols, obj.num_cols)
        return obj

    @classmethod
    def train(cls, transform):
        return cls._make_split(transform, "train")

    @classmethod
    def val(cls, transform):
        return cls._make_split(transform, "val")

    @classmethod
    def test(cls, transform):
        return cls._make_split(transform, "test")

    @classmethod
    def _make_split(cls, transform, split):
        obj = cls()
        obj.transform = transform

        (t19, v19, te19,
         t20, v20, te20) = make_splits(obj.df_2019, obj.df_2020)

        split_map = {"train": (t19, t20), "val": (v19, v20), "test": (te19, te20)}
        df19, df20 = split_map[split]

        df = pd.concat([df19, df20], ignore_index=True)
        meta = pd.concat([obj.df_2019_meta.loc[df19.index], obj.df_2020_meta.loc[df20.index], ], ignore_index=True)

        meta = obj.encode_meta(meta, obj.meta_cols, obj.num_cols)

        obj.df = df
        obj.meta = meta
        return obj

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]

        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)

        y = torch.tensor([float(row["MEL"])], dtype=torch.float32)

        if self.use_meta and self.meta is not None:
            m = torch.tensor(self.meta[i], dtype=torch.float32)
        else:
            m = torch.tensor([])

        return x, y, m

    def pathify(self, frame, image_col, split):
        return frame.assign(path="train/" + split + "/" + frame[image_col] + ".jpg")

    def encode_meta(self, frame, meta_cols, num_cols):

        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1)

        enc.fit(frame[meta_cols])
        cat = enc.transform(frame[meta_cols])

        num = frame[num_cols].to_numpy(np.float32)
        num = num / 100.0

        return np.concatenate([cat, num], axis=1).astype(np.float32)

    def get_df2019(self, csv_data="ISIC_2019_Training_GroundTruth.csv", csv_meta="ISIC_2019_Training_Metadata.csv"):
        assert csv_data.endswith("ISIC_2019_Training_GroundTruth.csv")
        assert csv_meta.endswith("ISIC_2019_Training_Metadata.csv")
        df_data = pd.read_csv(csv_data)
        df_meta = pd.read_csv(csv_meta)
        df = pd.merge(df_data, df_meta, on='image', how='left')
        cols = [
            "image", "sex", "age_approx", "anatom_site_general",
            "lesion_id", "MEL", "NV", "BCC", "AK", "BKL",
            "DF", "VASC", "SCC", "UNK"]
        missing = [c for c in cols if c not in df.columns]
        assert len(missing) == 0, f"Missing columns: {missing}\nCurrent columns: {df.columns}"
        df = df[cols].copy()
        return df, df_data.columns.tolist(), df_meta.columns.tolist()

    def get_df2020(self, path='ISIC_2020_Training_GroundTruth_v2.csv'):
        self.bkl = {'seborrheic keratosis', 'lentigo NOS', 'lichenoid keratosis', 'solar lentigo'}
        self.unknown = {'cafe-au-lait macule', 'atypical melanocytic proliferation'}
        df = self._get_df2020(path)

        df["diagnosis_collapsed"] = df["diagnosis"].apply(self.collapse_col)
        classes = ["MEL", "NV", "BKL", "DF", "VASC", "SCC", "AK", "BCC", "UNK"]

        # create binary columns
        for c in classes:
            df[c] = (df["diagnosis_collapsed"] == c).astype(int)

        # drop original labels
        df = df.drop(columns=["diagnosis", "diagnosis_collapsed"])
        return df

    def collapse_col(self, x):
        if x in self.bkl:
            return "BKL"
        if x in self.unknown:
            return "UNK"
        if x.lower() == "melanoma":
            return "MEL"
        if x.lower() == 'nevus':
            return "NV"
        if x.lower() == 'unknown':
            return "UNK"
        # nevus is sent to 0 in this class. Use class with nn.BCEwithlogits
        return "UNK"

    def _get_df2020(self, csv):
        df = pd.read_csv(csv)
        cols = ["image_name", "patient_id", "lesion_id", "sex", "age_approx", "anatom_site_general_challenge",
                "diagnosis", "benign_malignant", "target"]
        assert all(column in df.columns for column in
                   cols), f"train.csv is missing some columns. Current columns are: {df.columns}"
        df = df[cols].copy()
        return df

    def get_meta(self, df):
        if self.use_meta:
            df = df.copy()
            # Fill categorical missing values
            for c in self.meta_cols:
                df[c] = df[c].fillna("UNK")

            # Fill numeric missing values with median
            for c in self.num_cols:
                med = df[c].median()
                df[c] = df[c].fillna(med)
            return df
        else:
            return df

    def example_images(self, i):
        row = self.df.loc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)

        labels = row[self.label_cols].astype(float).to_numpy()
        y = torch.tensor(labels, dtype=torch.float32)

        if self.use_meta and self.meta is not None:
            m = torch.tensor(self.meta[i], dtype=torch.float32)
        else:
            m = torch.tensor([])
        # Convert transformed tensor back to image
        # x_pil = TF.to_pil_image(x)
        x_unnorm = unnormalize(x)  #

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Unnormalized Image")
        plt.imshow(x_unnorm)
        plt.axis("off")

        plt.show()
        return y, m


def unnormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # undo normalize
    t = t * std + mean
    # clamp to valid range
    t = t.clamp(0, 1)
    # convert to PIL
    return TF.to_pil_image(t)


def get_sampler(train_df):
    pos = (train_df["target"]==1).mean()
    w_pos = (1 - pos) / max(pos, 1e-6) # Define weights for data based on relative abundance of positives
    weights = train_df["target"].map({0:1.0, 1:w_pos}).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, w_pos

def multilabel_sampler(train_df, label_cols):
    # Count number of positives per class
    class_counts = train_df[label_cols].sum().replace(0, 1)

    # Inverse-frequency class weights
    class_weights = 1.0 / class_counts

    # Compute per-sample weight = sum(class_weights for labels that are 1)
    weights = (train_df[label_cols] * class_weights).sum(axis=1).values
    weights = weights.astype(float)

    gen = torch.Generator().manual_seed(42)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator = gen
    )
    return sampler, class_weights.to_dict()

# train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
# val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
# test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)