# Forschungspraxis: Adversarial Robustness of Deblurring Methods

Networks: RGDN and DWDN

## Requirements

Compatible with Python 3.8.10
Main requirements: PyTorch 1.7.1 is tested

## Evaluation

To evaluate the deep Wiener deconvolution network on test examples, run:
```eval
cd DWDM
python adversarial_entry.py
```

To evaluate the recurrent gradient descent network on test examples, run:
```eval
cd RGDN
python adversarial_entry.py
```
