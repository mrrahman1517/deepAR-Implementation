# deepar from scratch https://arxiv.org/abs/1704.04110
install:

cd deep_ar_library 
pip3 install -e .

train: 
cd .. python3 -m deep_ar_client.train --data data/Hourly-train.csv --epochs 10 --checkpoint model.pt

validate:

python3 -m deep_ar_client.validate --checkpoint model.pt --data data/Hourly-train.csv --plot
# deepAR-Implementation
