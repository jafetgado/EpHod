**EpHod**
===============

EpHod is a deep learning model to predict the optimum pH of enzymes (pHopt). 
EpHod uses a  light-attention top model with 49 million parameters on the 
protein language model, ESM-1v, with 690 million parameters, and was trained 
on 1.9 million proteins with optimal environment pH (pHenv) followed by 9,855 
enzymes with of catalytic optimum pH (pHopt). 

Dependencies are in `env.yml`, and we recommend using a conda environment.

Weights of EpHod model and training datasets are available at `Zenodo <https://doi.org/10.5281/zenodo.8011249>`__.




Usage 
-------------

1. Clone repository and install conda environment. Installation with the required environment takes roughly four minutes.

.. code:: shell-session

    git clone https://github.com/jafetgado/EpHod.git
    cd EpHod
    export PYTHONPATH="$(pwd)"
    conda env create -f ./env.yml -p ./env
    conda activate ./env
..
    	
	
2. Predict pHopt with EpHod language (needs gpu). Predicted output is a csv file with accession codes and predicted pHopt in columns.

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
  
    
3. Alternatively (in place of No. 2), predict with a traditional regression model (support vector regression with amino acid composition).
This is less accurate but a very fast estimation for wild type enzymes with CPU

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

Gado J.E., Knotts M., Shaw A.Y., et al, 2023. "Deep learning prediction of enzyme optimum pH".
