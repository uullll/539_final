This project develops a melanoma detector using CNN-based architectures. Our project is inspired by the 2020 SIIM-ISIC competition. See https://www.kaggle.com/competitions/siim-isic-melanoma-classification for more details.

## Data
Datasets are available for download at https://challenge2020.isic-archive.com/.
 
## Environment
- Python >= 3.8
- Recommended: conda / venv

Install dependencies:
```bash
pip install -r requirements.txt
```
## Project Structure
project-root/
├── Exported_Figures/                      # Saved plots and evaluation figures
│   ├── EffNetB6 Confusion Matrix.jpg
│   ├── EffNetB6 Probability Histogram.jpg
│   ├── ResNet101 with Weighted Loss Confusion Matrix.jpg
│   └── ResNet101 with Weighted Loss Probability Histogram.jpg
├── src/                                   # Core source code
│   ├── 2019+2020/                         # Combined 2019–2020 experiments
│   │   ├── Dataset_2019_2020.py           # Dataset definition
│   │   ├── Models_lesion.py               # Model architectures
│   │   ├── Train.py                       # Training pipeline
│   │   ├── Evaluate_multiclass.py         # Evaluation logic
│   │   └── Utility.py                     # Helper functions
│   ├── 2020/                              # 2020-only experiments
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── 2020_Patient/                      # Patient-level dataset handling
│       └── PatientDataset.py
├── app.py                                 # Application / inference entry point
├── main.html                              # Frontend or visualization page
├── main_2020.ipynb                        # 2020 experiment notebook
├── main_2020_2019_BCE.ipynb               # 2019+2020 BCE training notebook
├── requirements.txt                       # Python dependencies
├── sample_submission.csv                  # Submission template
└── README.md                              # Project documentation


## Set up
First run all in main_2020.ipynb, main will call the fuction from the api in src. After running, main will create model like Best_Efficient_net_B6.pt, Best_Restnet101_meta.pt, best.pt.


## Run
Then you can run:


```bash
python3 app.py
```
It will create a website to analyze the image.
