**EpHod**
===============

EpHod is a deep semi-supervised language model to predict the optimum pH of
enzymes (pHopt).

Usage 
-------------
.. code:: shell-session

    git clone https://github.com/jafetgado/EpHod.git
    cd EpHod
    conda env create -f ./env.yml -p ./env
    conda activate ./env

    # Predict pHopt with EpHod language (needs gpu)
    
    python ./ephod/runner.py \
        --fasta_path "./example/sequences.fasta" \
        --save_dir ./example \
        --batch_size 8 \
        --verbose 1 \
        --save_attention_weights 1 \
        --save_embeddings 1 
    
    # Or predict with a simple learning regression model 
    # (support vector regression with amino acid composition)
    # Less accurate but a very fast estimation for wild type enzymes with CPU
	
    python ./ephod/runner.py \
        --fasta_path "./example/sequences.fasta" \
        --save_dir ./example \
        --aac_svr 1 \
        --verbose 1 
..



Citation
----------
If you find EpHod useful, please cite the following:

Gado JE, Shaw AY, et al, 2023. "Predicting enzyme optimum pH with deep language models".
