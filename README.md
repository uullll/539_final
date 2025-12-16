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

```text
project-root/
├── Exported_Figures/                      # Saved figures and plots from model evaluations
├── src/                                   # Core source code
│   ├── 2019+2020/                         # Combined 2019 and 2020 dataset experiments
│   │   ├── Dataset_2019_2020.py           # Dataset definitions for 2019+2020
│   │   ├── Models_lesion.py               # Model architectures
│   │   ├── Train.py                       # Training scripts
│   │   ├── Evaluate_multiclass.py         # Evaluation scripts for multi-class classification
│   │   └── Utility.py                     # Helper functions
│   ├── 2020/                              # 2020-only experiments
│   │   ├── dataset.py                      # Dataset definitions for 2020
│   │   ├── models.py                       # Model architectures for 2020
│   │   ├── train.py                        # Training scripts
│   │   └── evaluate.py                     # Evaluation scripts
├── app.py                                 # Main application 
├── main.html                              # Frontend or visualization page
├── main_2020.ipynb                        # Notebook for 2020 experiments
├── main_2020_2019_BCE.ipynb               # Notebook for combined 2019+2020 BCE training
├── main_2020_2019_BCE.ipynb               # Notebook for patient-level classifier
├── requirements.txt                       # Python dependencies
├── sample_submission.csv                  
└── README.md                              # Project documentation
```


## Set up
First run all in main_2020.ipynb, main will call the fuction from the api in src. After running, main will create model like Best_Efficient_net_B6.pt, Best_Restnet101_meta.pt, best.pt.


## Run
Then you can run:


```bash
python3 app.py
```
It will create a website to analyze the image.
