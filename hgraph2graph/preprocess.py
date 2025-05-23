from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy

from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    try:
        return MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    except Exception as e:
        print(f"Error processing batch: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu)
    try:
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]

        random.shuffle(data)

        batches = [data[i: i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize, vocab=args.vocab)
        all_data = pool.map(func, batches)
        all_data = [d for d in all_data if d is not None]  # Filter out invalid batches

        if len(all_data) == 0:
            print("No valid data found. Exiting.", file=sys.stderr)
            sys.exit(1)  # Exit the script if no valid data is found

        num_splits = len(all_data) // 1000
        if num_splits == 0:
            num_splits = 1  # Ensure at least one split to avoid division by zero

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st: st + le]

            with open(f'tensors-{split_id}.pkl', 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
    finally:
        pool.close()
        pool.join()