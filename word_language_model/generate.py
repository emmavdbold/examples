###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse

import torch

import data

import random

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--input', type=str, default=None,
                    help='first word(s) of the text to be generated')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

if args.input:
    
    assert not is_transformer_model, "Input prompts are currently not supported for Transformer models."
    input_list = args.input.split()
    get_words_not_in_vocab = lambda word_list : [word for word in word_list if word not in corpus.dictionary.word2idx]
    words_not_in_vocab = get_words_not_in_vocab(input_list)

    while words_not_in_vocab:
        words_not_in_vocab = ', '.join(words_not_in_vocab)
        sample_words = ', '.join(random.sample(['"' + word + '"' for word in list(corpus.dictionary.word2idx.keys())], 5))
        new_input = input(f"These words are not part of the vocabulary: {words_not_in_vocab}.\n\
            Here are a few suggestions: {sample_words}.\n\
            Type in another input prompt: ")
        input_list = new_input.split()
        words_not_in_vocab = get_words_not_in_vocab(input_list)
    
    input_length = len(input_list)

else:
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                if args.input and i < input_length:
                    # tensor([[word_index]])
                    input = torch.tensor(corpus.dictionary.word2idx[input_list[i]]).reshape((1, 1)).to(device)
                    # pass input through model -- hidden updated each time
                    output, hidden = model(input, hidden)
                    # store index of word in new variable for use below    
                    word_idx = input
                else:
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
