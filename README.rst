**EpHod**
===============

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15015124.svg
   :target: https://doi.org/10.5281/zenodo.15015124
   :alt: DOI

EpHod is a deep-learning model to predict the optimum pH of enzymes (pHopt). The model is of an ensemble of a neural network (residual light attention or RLAT) and a support vector regression (SVR) model both trained on top of ESM-1v embeddings. The neural network (RLATtr) was first pretrained using 1.9 million proteins with optimal environment pH (pHenv) labels, followed by fine tuning using 9,855 enzyme with catalytic optimum pH labels (pHopt).

We recommend using a conda environment. Dependencies are in `env.yml`. The code was successfully run with PyTorch v1.7.0 and CUDA v 11.7.
Weights of EpHod model and training datasets are available at `Zenodo <https://doi.org/10.5281/zenodo.14252615>`__.



Usage 
-------------

1. Clone repository and install conda environment. Installation with the 
required environment takes roughly four minutes.

.. code:: shell-session

    git clone https://github.com/jafetgado/EpHod.git
    cd EpHod
    conda env create -f ./env.yml -p ./env
    conda activate ./env
..
    	
2. Predict pHopt with EpHod. Predicted pHopt values, and attention weights from the RLATtr model, as well as the embeddings from the final RLATtr layer (2560-dim) are saved in ``./example/``. Pass 0 to ``--save_attention_weights`` 
and ``--save_embeddings`` to avoid writing the weights and embeddings output. 
Besides downloading model weights, which may take several minutues, with a batch size of 1, prediction takes ~7 seconds/sequence on a CPU and ~0.1 seconds/sequence on a GPU.

.. code:: shell-session

    python ./ephod/run.py \
        --fasta_path "./example/test_sequences.fasta" \
        --save_dir ./example \
        --csv_name prediction.csv \
        --verbose 1 \
        --save_attention_weights 0 \
        --save_embeddings 0 
..




Citation
----------
If you find EpHod useful, please cite the following:

Gado J.E., Knotts M., Shaw A.Y., et al, 2024. "Machine learning prediction of enzyme optimum pH". `Nature Machine Intelligence <https://doi.org/10.1038/s42256-025-01026-6>`__.

