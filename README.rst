**EpHod**
===============

EpHod is a deep-learning model to predict the optimum pH of enzymes (pHopt). 
The model architecture consists of  a light-attention module with 49 million parameters 
on top of the protein language model, ESM-1v, with 690 million parameters. EpHod 
was trained on 1.9 million proteins with optimal environment pH (pHenv) followed 
by 9,855 enzymes with of catalytic optimum pH (pHopt). 

We recommend using a conda environment. 

Dependencies are in `env.yml`.

Weights of EpHod model and training datasets are available at `Zenodo <https://doi.org/10.5281/zenodo.8011249>`__.




Usage 
-------------

1. Clone repository and install conda environment. Installation with the 
required environment takes roughly four minutes.

.. code:: shell-session

    git clone https://github.com/jafetgado/EpHod.git
    cd EpHod
    export PYTHONPATH="$(pwd)"
    conda env create -f ./env.yml -p ./env
    conda activate ./env
..
    	
	
2. Predict pHopt with EpHod (needs gpu). Predicted pHopt values, and 
softmax weight values (attention weights), as well as final EpHod layer embeddings 
(2560-dim) are saved in ``./example/``.Pass 0 to ``--save_attention_weights`` 
and ``--save_embeddings`` to avoid writing the weights and embeddings output. 
Besides downloading model weights, which may take several minutues, with a batch 
size of 1, prediction takes ~7 seconds/sequence on a CPU and ~0.1 seconds/sequence 
on a GPU.

.. code:: shell-session

    python ./ephod/predict.py \
        --fasta_path "./example/sequences.fasta" \
        --save_dir ./example \
        --csv_name ephod_pred.csv \
        --batch_size 1 \
        --verbose 1 \
        --save_attention_weights 1 \
        --attention_mode "average" \
        --save_embeddings 1 
..
  
    
3. Alternatively, predict pHopt with a support vector regression model 
based on the amino acid composition (AAC-SVR). This may be less accurate 
but is a very fast estimation with CPU.

.. code:: shell-session

    python ./ephod/predict.py \
        --fasta_path "./example/sequences.fasta" \
        --save_dir ./example \
        --csv_name svr_pred.csv \
        --aac_svr 1 \
        --verbose 1 
..



Citation
----------
If you find EpHod useful, please cite the following:

Gado J.E., Knotts M., Shaw A.Y., et al, 2023. "Deep learning prediction of enzyme optimum pH". `bioRxiv <https://www.biorxiv.org/content/10.1101/2023.06.22.544776v1.abstract>`__.
