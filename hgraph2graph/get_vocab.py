import sys
import argparse
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        try:
            # Attempt to create MolGraph
            hmol = MolGraph(s)
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                vocab.add(attr['label'])
                for i, s in attr['inter_label']:
                    vocab.add((smiles, s))
        except Exception as e:
            # Print errors to stderr
            print(f"Error processing SMILES: {s} — {e}", file=sys.stderr)
            continue
    return vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # Read input data from stdin
    data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))  # Remove duplicates

    # Split data into batches for multiprocessing
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    # Create a pool of processes
    pool = Pool(args.ncpu)
    try:
        vocab_list = pool.map(process, batches)
    finally:
        pool.close()
        pool.join()

    # Combine vocabularies from all processes
    vocab = set()
    for v in vocab_list:
        vocab.update(v)

    # Output the vocabulary to stdout
    for item in sorted(vocab):
        if isinstance(item, tuple):
            print(f"{item[0]} {item[1]}")
        else:
            print(item)