"""
Train a neural network model with distributed training
"""


import numpy as np
import pandas as pd
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp


import os
import subprocess
import importlib
import sys
sys.path.append('./training/')
import trainutils
import nn_models
importlib.reload(trainutils);




def create_parser():
    '''Parse command-line training arguments'''
    
    parser = argparse.ArgumentParser(description="Train NN mode")
    parser.add_argument('--trainseqs', type=str, 
                        help='File containing accession codes of training sequences in csv format')
    parser.add_argument('--valseqs', type=str,
                        help='File containing accession codes of validation sequences in csv format')
    parser.add_argument('--target_data', type=str,
                        help='Path to csv file containing target labels and sample weights')
    parser.add_argument('--embedding_dir', type=str,
                        help='Path to directory containing per-residue embeddings of sequences')
    parser.add_argument('--model_name', type=str, 
                        help='Name of model used in saving parameters.')                             
    parser.add_argument('--paramsjson', type=str,
                        help='Json file containing hyperparameters for building and training NN model')
    parser.add_argument('--savedir', type=str, 
                        help='Directory to save model parameters and training progress')
    parser.add_argument('--distributed', type=int,
                        help='Whether to train with a single GPU (0) or multiple GPUs(1)')
    parser.add_argument('--maxlen', default=1024, type=int,
                        help='Maximum length of sequences. Embeddings will be post-padded to this length with zeros')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size of single GPU used in training and inference (i.e. local, non-distributed)')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='Number of nodes for distributed training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='The maximum number of epochs for training')
    parser.add_argument('--reduce_lr_patience', default=20, type=int, 
                        help='Reduce learning rate by 0.5 if validation loss does not improve after reduce_lr_patience epochs')
    parser.add_argument('--stop_patience', default=100, type=int,
                        help='Exit training if validation loss does not improve after top_patience epochs')
    parser.add_argument('--restart', default=0, type=int,
                        help='Whether to begin new training (0) and overwrite checkpoint files, or to restart training (1) and append checkpoint files')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Whether to print out details of training (1) or not (0)')
    args = parser.parse_args()
    args.params = utils.read_json(args.paramsjson)  # Model parameters as dict

    return args




def configure_cuda(args):
    '''Configure devices for training'''
    
    if not torch.cuda.is_available():
        print("FATAL: GPU not available")
        exit()
    torch.backends.cudnn.benchmark = True  # For efficient training
    args.gpus_per_node = torch.cuda.device_count()   
    args.num_cpus = os.cpu_count()     
    if args.distributed:
        args.world_size = args.gpus_per_node * args.num_nodes
    else:
        args.world_size = 1
    
    return args.world_size, args




def configure_processes(rank, args):
    '''Setup the processes group for distributed training'''
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)




def get_dataset(seqs, sample_method, args):
    '''Get dataset of embeddings data'''

    accessions = pd.read_csv(seqs, index_col=0).iloc[:,-1].values # Seq. accession codes
    target_data = pd.read_csv(args.target_data, index_col=0)
    dataset = dataproc.EmbeddingData(
        accessions=accessions, 
        y=target_data.loc[accessions,'y'].values,
        weights=target_data.loc[accessions, sample_method].values, 
        embedding_dir=args.embedding_dir, 
        use_mask=True, 
        maxlen=args.maxlen
        )
    
    return dataset




def get_distributed_dataloader(rank, world_size, dataset, train_mode, args):
    '''Return a dataloader for distributed training/inference'''
    
    sampler = DistributedSampler(dataset, 
                                 num_replicas=world_size, 
                                 rank=rank,
                                 shuffle=train_mode, 
                                 drop_last=train_mode)
    dataloader = DataLoader(dataset, 
                            sampler=sampler,
                            batch_size=args.batch_size,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            drop_last=train_mode)
    
    return dataloader    
    
    
    

def build_model(args):
    '''Build neural network model'''

    # Build model with nn.Module
    args.model_params = {
        key: args.params[key] for key in 
        ['dim', 'dim', 'kernel_size', 'dropout', 'activation', 'res_blocks', 
         'random_seed']
        }
    torch.manual_seed(args.model_params['random_seed'])  # Reproducibility
    model = nn_models.ResidualLightAttention(**args.model_params) # Replace as needed.
    args.num_parameters = torchmodels.count_parameters(model)['FULL_MODEL']
    
    return model, args




def build_optimizer(model, args):
    '''Build optimizer for model'''
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.params['learning_rate'],
                           weight_decay=args.params['l2reg'])
    
    return optimizer
    

    

def save_model(model, optimizer, path, args):
    '''Save model and optimizer parameters to disk'''
    
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_params': args.model_params,
            'optimizer_state_dict': optimizer.state_dict(),
            },
        path)    




def load_model(model, path, args):
    '''Load model parameters from path'''
    
    checkpoint = torch.load(path, map_location='cpu')
    assert args.model_params == checkpoint['model_params'], \
        "Saved model params and current params are different"
    '''
    model_state_dict = {k.partition('model.')[]: v for k,v in \
                        checkpoint['model_state_dict']}
    '''
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict, strict=False)
    
    return model    
    
    


def load_optimizer(optimizer, path, args):
    '''Load model parameters from path'''
    
    checkpoint = torch.load(path, map_location='cpu')
    assert args.model_params == checkpoint['model_params'], \
        "Saved model params and current params are different"
    '''
    optimizer_state_dict = {k.partition('model.')[2]: v for k,v in \
                            checkpoint['optimizer_state_dict']}
    '''
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    optimizer.load_state_dict(optimizer_state_dict)
    
    return optimizer




def reduce_learning_rate(optimizer):
    '''Reduce learning rate by a factor of 0.5'''

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
    
    return optimizer




def root_mean_squared_error(ytrue, ypred, weight):
    '''Return the weighted mean squared error'''
    
    return torch.sqrt(torch.mean(((ytrue - ypred) ** 2) * weight))
  



def prepare_training(args):
    '''Prepare paths and objects for training'''
    
    # Paths/directories
    args.savepath = f'{args.savedir}/{args.model_name}'
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    args.recent_model = f'{args.savepath}/model_recent.pt' # Save model each epoch
    args.best_model = f'{args.savepath}/model_best.pt'     # Save model if error decreases
    args.log_file = f'{args.savepath}/log.csv'             # Write losses to file
    
    
    # Instantiate model 
    # Don't instantiate optimizer until model.to(device) is called
    model, args = build_model(args)
    
    
    # Model and training progress parameters
    if (not args.restart): # Train from scratch, don't restart
        
        # Write new log file
        with open(args.log_file, 'w') as logs:
            logs.write("epoch,train_loss,val_loss,best_val_loss,learning_rate,"\
                       "since_improved\n")
        
        # Progress parameters
        progress_params = {'epoch': 1, # Start from 1 to args.epoch + 1, not from 0
                           'best_val_loss': np.nan,
                           'since_improved': 0}
            
    else: # Restart training, don't train from scratch
        
        # Don't write new log files, assert that necessary files are in savepath
        error_msg = "Cannot restart training without "
        assert os.path.exists(args.log_file), error_msg + "progress file"
        assert os.path.exists(args.best_model), error_msg + "best_model"
        
        # Get progress parameters from log file
        logs = pd.read_csv(args.log_file, index_col=False, header=0)
        progress_params = {'epoch': logs['epoch'].values[-1],
                           'best_val_loss': logs['best_val_loss'].values[-1], 
                           'since_improved': logs['since_improved'].values[-1]}        

        # Load model from checkpoint
        model = load_model(model, args.best_model, args)
        
    return model, progress_params, args
     



def start_message(args):
    '''Description message at start of training'''
    
    if (args.verbose):
        print(f'INFO: MODEL_NAME = {args.model_name}')
        print(f'INFO: NUM_PARAMETERS = {args.num_parameters:.2e}')
        print(f'INFO: EPOCHS = {args.epochs}')
        print(f'INFO: NUM_NODES = {args.num_nodes}')
        print(f'INFO: GPUS_PER_NODE = {args.gpus_per_node}')
        print(f'INFO: NUM_GPUS = {args.world_size}')
        print(f'INFO: NUM_CPUS = {args.num_cpus}')
        print(f'INFO: NUM_WORKERS = {args.num_workers}')        
        print(f'INFO: SAVE_PATH = {args.savepath}')
        print()
        for key, value in args.params.items():
            print(f'PARAMS: {key}={value}')
        print()
        print('INFO: STARTING TRAINING')
        print()
        



def exit_message(epoch, args):
    '''Exit message at end of training'''
    
    if epoch >= args.epochs:
        print()
        print(f'INFO: Maximum training epochs ({args.epochs}) reached!')
        print('INFO: FINISHED TRAINING')
        
        return
    
    


def train_for_one_epoch(rank, trainloader, model, optimizer, epoch, args):
    '''Train model for one epoch'''
    
    _ = model.train()  # Set in training mode
    losses = []

    for i, (x, y, weight, mask) in enumerate(trainloader):

        x, y, weight, mask = x.to(rank), y.to(rank), weight.to(rank), mask.to(rank)
        optimizer.zero_grad()                          
        ypred = model(x, mask)[0]
        loss = root_mean_squared_error(y, ypred, weight) 
        _ = loss.backward()
        _ = optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)




def validate(rank, valloader, model, args):
    '''Validate model on validation set'''
    
    _ = model.eval()  # Set in validation mode
    losses = []
    
    with torch.no_grad():
        
        for i, (x, y, weight, mask) in enumerate(valloader):
            
            x, y, weight, mask = x.to(rank), y.to(rank), weight.to(rank), mask.to(rank)
            ypred = model(x, mask)[0]
            loss = root_mean_squared_error(y, ypred, weight)
            losses.append(loss.item())
            
    return np.mean(losses)
            



def distributed_training(rank, world_size, args):
    '''Distributed training routine'''

    # Set up process groups for distributed training/inference
    _ = configure_processes(rank, args)
    
    
    # Prepare dataloader
    traindata = get_dataset(args.trainseqs, 
                            args.params['sample_method'], 
                            args)
    trainloader = get_distributed_dataloader(rank, 
                                             world_size,
                                             traindata,
                                             True,
                                             args)
    valdata = get_dataset(args.valseqs, 
                          'bin_inv', # Use bin_inv to reweight validataion data
                          args)
    valloader = get_distributed_dataloader(rank,
                                           world_size,
                                           valdata, 
                                           False,
                                           args)
    
    # Prepare paths and model objects for training
    model, progress_params, args = prepare_training(args)
    model = model.to(rank)  # Move model to GPU device in DDP
    if rank == 0:
        start_message(args)

    
    # Build optimizer 
    optimizer = build_optimizer(model, args)
    if args.restart:
        optimizer = load_optimizer(optimizer, args.best_model, args)
    

    # Wrap the model as a DistributedDataParallel object
    model = DistributedDataParallel(model, 
                                    device_ids=[rank],
                                    output_device=rank,
                                    find_unused_parameters=True)
    
    
    # Training epochs
    start_epoch = progress_params['epoch']  # start_epoch >0 if restarting
    stop_epoch = args.epochs + 1
    if rank == 0:
        _ = exit_message(start_epoch, args)
    
    # Training iteration
    for epoch in range(start_epoch, stop_epoch):
        
        _ = trainloader.sampler.set_epoch(epoch) # For distributed batches
        learning_rate = optimizer.param_groups[0]['lr']
        start_time = time.time()
        train_loss = train_for_one_epoch(rank, trainloader, model, optimizer,
                                                    epoch, args)
        val_loss = validate(rank, valloader, model, args)
        end_time = time.time()
        total_time = end_time - start_time 

        
        # Writing and priting, only if gpu is master
        if rank == 0:
            
            # Save model and update progress parameters
            _ = save_model(model, optimizer, args.recent_model, args)
            
            if not (val_loss >= progress_params['best_val_loss']):
                # Save model as best model, if validation loss has improved
                _ = save_model(model, optimizer, args.best_model, args)
                progress_params['best_val_loss'] = val_loss
                progress_params['since_improved'] = 0 # Reset
            else:
                progress_params['since_improved'] += 1 
                
            
            # Reduce learning rate if performance hasn't improved
            if (progress_params['since_improved']>= args.reduce_lr_patience) and \
                (progress_params['since_improved'] % args.reduce_lr_patience == 0):
                    optimizer = reduce_learning_rate(optimizer)
                
            # Log progress
            with open(args.log_file, 'a') as logs:
                logs.write(f"{epoch},{train_loss},{val_loss},"\
                           f"{progress_params['best_val_loss']},"\
                           f"{learning_rate},"\
                           f"{progress_params['since_improved']}\n")
                    
            if args.verbose == 1:
                print(f'PROGRESS: epoch={epoch}, '\
                      f'train_loss={train_loss:.4f}, '\
                      f'val_loss={val_loss:.4f}, '\
                      f"best_val_loss={progress_params['best_val_loss']:.4f}, "\
                      f"learning_rate={learning_rate:.2e}, "\
                      f"since_improved={progress_params['since_improved']}, "\
                      f"time={total_time:.1f}s")
            
            # Exit if validation hasn't improved 
            if progress_params['since_improved'] >= args.stop_patience:
                print(f"INFO: Exiting, as validation loss has not improved since "\
                      f"{progress_params['since_improved']} epochs")
                print("INFO: FINISHED TRAINING")
                return
                
    
    # Exit training
    if rank == 0:
        _ = exit_message(epoch, args)
    return




def main():
    '''Main routine'''
    
    # Parse command-line training arguments
    args = create_parser()                
    
    # Configure GPUs for processes
    world_size, args = configure_cuda(args)        
    
    # Distributed training raining routine
    _ = mp.spawn(distributed_training,
                 nprocs=args.world_size,
                 args=(args.world_size, args)
                 )




if __name__=='__main__':
    _ = main()