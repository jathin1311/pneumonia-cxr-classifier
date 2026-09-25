"""Pneumonia vs. normal classification on PneumoniaMNIST pediatric chest X-rays.

Modules, in the order data flows through them:

    data.py      load and check the dataset, build PyTorch data loaders
    models.py    the two models: logistic regression and a small CNN
    train.py     fit both models (the CNN with a hand-written training loop)
    evaluate.py  threshold choice, metrics, and bootstrap confidence intervals
    plots.py     every figure the project produces
    utils.py     seeding, device choice, saving files, environment info
    cli.py       the `pcxr` command that ties everything together
"""

__version__ = "0.1.0"
