import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("-c", "--corpus_files", nargs='+', type=str,
                    default=["data/corpus/elmo.corpus.small.txt"])
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("--verbose_freq", type=int, default=100)

parser.add_argument("--word_vocab_path", type=str, default="data/vocab/word.90k.vocab")
parser.add_argument("--char_vocab_path", type=str, default="data/vocab/jamo.100.vocab")

parser.add_argument("--word_seq_len", type=int, default=10)
parser.add_argument("--char_seq_len", type=int, default=10)

parser.add_argument("--char_embedding_dim", type=int, default=64)
parser.add_argument("--kernel_sizes", nargs='+', type=int, default=[1, 2, 3])
parser.add_argument("--filter_sizes", nargs='+', type=int, default=None)

parser.add_argument("--elmo_hidden", type=int, default=128)
parser.add_argument("--steps_per_epoch", type=int, default=128)

args = parser.parse_args()

config_dict = vars(args)

print(config_dict)