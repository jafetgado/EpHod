**EpHod**
===============

EpHod is a deep semi-supervised language model to predict the optimum pH of
enzymes (pHopt).

Usage 
-------------

1. Clone repository and install conda environment

.. code:: shell-session

    git clone https://github.com/jafetgado/EpHod.git
    cd EpHod
    export PYTHONPATH="$(pwd)"
    conda env create -f ./env.yml -p ./env
    conda activate ./env
..
    	
	
2. Predict pHopt with EpHod language (needs gpu)

.. code:: shell-session

    python ./ephod/predict.py \
        --fasta_path "./example/sequences.fasta" \
        --save_dir ./example \
        --csv_name predictions.csv \
        --batch_size 8 \
        --verbose 1 \
        --save_attention_weights 0 \
        --attention_mode "None" \
        --save_embeddings 0 
..
  
    
3. Alternatively (in place of No. 2), predict with a traditional regression model (support vector regression with amino acid composition).
This is less accurate but a very fast estimation for wild type enzymes with CPU

.. code:: shell-session

    python ./ephod/predict.py \
        --fasta_path "./example/sequences.fasta" \
        --save_dir ./example \
        --csv_name prediction.csv \
        --aac_svr 1 \
        --verbose 1 
..



Citation
----------
If you find EpHod useful, please cite the following:

Gado J.E., Knotts M., Shaw A.Y., et al, 2023. "Deep learning prediction of enzyme optimum pH".
