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
├── Exported_Figures/
│   ├── EffNetB6 Confusion Matrix.jpg
│   ├── EffNetB6 Probability Histogram.jpg
│   ├── ResNet101 with Weighted Loss Confusion Matrix.jpg
│   └── ResNet101 with Weighted Loss Probability Histogram.jpg
├── src/
│   ├── 2019+2020/
│   │   ├── Dataset_2019_2020.py
│   │   ├── Models_lesion.py
│   │   ├── Train.py
│   │   ├── Evaluate_multiclass.py
│   │   └── Utility.py
│   ├── 2020/
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── 2020_Patient/
│       └── PatientDataset.py
├── app.py
├── main.html
├── main_2020.ipynb
├── main_2020_2019_BCE.ipynb
├── requirements.txt
├── sample_submission.csv
└── README.md



## Set up
First run all in main_2020.ipynb, main will call the fuction from the api in src. After running, main will create model like Best_Efficient_net_B6.pt, Best_Restnet101_meta.pt, best.pt.


## Run
Then you can run:


```bash
python3 app.py
```
It will create a website to analyze the image.
