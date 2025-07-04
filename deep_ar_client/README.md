# deep_ar_client

Lightweight example client that **uses** the modular `deep_ar` library
to train and forecast on the [M4 forecasting dataset](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset).

## Quickstart

```bash
# 1. Clone / install deep_ar (the library)
pip install -e ../deep_ar       # assuming sibling folder

# 2. Install client requirements
pip install -r requirements.txt

# 3. Train
python train.py --data /path/to/Hourly-train.csv --epochs 10 --checkpoint model.pt

# 4. Forecast one series & show plot
python forecast.py --checkpoint model.pt --data /path/to/Hourly-train.csv --index 0
```
