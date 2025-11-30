This project develops a patient-level melanoma detector using CNN-based architectures. Our project is inspired by the 2020 SIIM-ISIC competition. See https://www.kaggle.com/competitions/siim-isic-melanoma-classification for more details.

## Data
Datasets are available for download at https://challenge2020.isic-archive.com/.

## Updates Nov 30
I finished the patient level classifer (see PatientDataset file). I am using MIL based pooling method from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/main.py and a cascade loss method from https://github.com/svishwa/crowdcount-cascaded-mtl/blob/master/src/crowd_count.py. I am not including my code for MIL based pooling and cascade loss because so far I have had no success. I need to revise the implementation. 
