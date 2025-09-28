from collections import Counter

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=[]):
        if tokens is None:
            tokens = []
        
        # 统计词频
        counter = Counter(tokens)
        self.min_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 构建词表
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 加入词频大于等于min_freq的词
        for token, freq in self.min_freq:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)
    
def numericalize(tokens: list, vocab: Vocab):
    """
    将词元列表转换为数字列表,并同步返回词表
    :param tokens: 词元列表
    :param vocab: 词表
    :return: 数字列表，词表
    """
    indices = []
    for token in tokens:
        if token in vocab.token_to_idx:
            indices.append(vocab.token_to_idx[token])
        else:
            indices.append(vocab.token_to_idx['<unk>'])
    return indices, vocab

if __name__ == '__main__':
    import data_preporcess

    file_path = r'.\\data\\time-machine-data.txt'
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        file = f.read()
    text = data_preporcess.data_clean_for_time_machine(file)
    tokens = data_preporcess.tokenize(text)
    vocab = Vocab(tokens, min_freq=2, reserved_tokens=['<pad>', '<unk>'])
    indices, vocab = numericalize(tokens, vocab)
    print(indices[:10])
    print(vocab.idx_to_token[:10])


