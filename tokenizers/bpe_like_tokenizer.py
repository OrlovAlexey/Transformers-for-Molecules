from collections import Counter, deque
from functools import lru_cache
import json


class BPETokenizerSimple:
    def __init__(self):
        self.vocab = {}  # maps token_id to token_str
        self.inverse_vocab = {}# maps token_str to token_id
        self.bpe_merges = {} # example: {(token_id1, token_id2): merged_token_id}

    def train(self, text, vocab_size, special_tokens=None):
        """
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            special_tokens (set): A set of special tokens to include.
        """
        text_splitted_by_space = text.split(" ")
        text_splitted_by_space = list(filter(lambda a: a != "", text_splitted_by_space)) # remove ""
        
        # get all possible symbols/'chars'
        unique_symbols_list = ["."]
        for symbol in text_splitted_by_space:
            if symbol == "." or symbol == "" or symbol == " ":
                continue
            unique_symbols_list.append(symbol)
            
        unique_symbols = set(unique_symbols_list)

        self.vocab = {i: symbol for i, symbol in enumerate(unique_symbols)}
        self.inverse_vocab = {symbol: i for i, symbol in self.vocab.items()}

        # add special tokens
        if special_tokens:
            for token in special_tokens:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # tokenize all text
        token_ids = [self.inverse_vocab[symbol] for symbol in text_splitted_by_space]

        # find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            max_freq_pair = self.find_freq_pair(token_ids, mode="most")
            if max_freq_pair is None:  # no more pairs to merge
                break
            token_ids = self.replace_pair(token_ids, max_freq_pair, new_id)
            # print(token_ids)
            self.bpe_merges[max_freq_pair] = new_id

        # add merged tokens to vocabulary 
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + " " + self.vocab[p1] # "Ġ"
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    @staticmethod
    def get_counts_of_pairs(token_ids):
        return Counter(zip(token_ids, token_ids[1:]))

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if len(pairs.items()) == 0:
            return None
        
        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, max_freq_pair, new_id):
        """
            Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == max_freq_pair:
                replaced.append(new_id)
                dq.popleft() # remove the second token of pair
            else:
                replaced.append(current)

        return replaced

    def encode(self, text):
        """
            text (str): The text to encode.
        """
        
        #words = text.replace("\n", " \n ").split() 
        text_splitted_by_space = text.split(" ")
        token_ids = [self.inverse_vocab[symbol] for symbol in text_splitted_by_space]
        
        while len(token_ids) >= 2:
            # find the pair with the lowest merge index
            stats = self.get_counts_of_pairs(token_ids)
            
            pair = min(stats, key=lambda p: self.bpe_merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.bpe_merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.bpe_merges[pair]
            token_ids = self.replace_pair(token_ids, pair, idx)
        return token_ids

    def decode(self, token_ids):
        """
            token_ids (List[int]): The list of token IDs to decode.
        """
        decoded_string = ""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            decoded_string += token
            decoded_string += " "
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump({k: v for k, v in self.vocab.items()}, file, ensure_ascii=False, indent=2)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge['pair'])
                new_id = merge['new_id']
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    