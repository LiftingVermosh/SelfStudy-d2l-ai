# 对 ../data 下的time-machine-data.txt文件进行预处理。

def data_clean_for_time_machine(text):
    """
    清理文本,只针对《时间机器》这本书
    """
    if not isinstance(text, str):
        raise TypeError("text should be a string")
    start_idx = text.find(r"*** START OF THE PROJECT GUTENBERG EBOOK 35 ***")
    end_idx = text.find(r"*** END OF THE PROJECT GUTENBERG EBOOK 35 ***")
    if start_idx == -1 or end_idx == -1:
        raise ValueError("text is not a valid Project Gutenberg book")
    text = text[start_idx:end_idx]
    lines = []
    for line in text.split("\n"):
        if not line.strip() or line.strip().isupper():
            continue
        lines.append(line.strip())
    return " ".join(lines).lower()  # 合并为单字符串并转化为小写

def tokenize(text):
    """
    文本分词：将连续的文本序列切分成一个个词元
    """
    if not isinstance(text, str):
        raise TypeError("text should be a string")
    return text.split()

def char_tokenize(text):
    """
    文本分词：将文本按字符切分成一个个词元
    """
    if not isinstance(text, str):
        raise TypeError("text should be a string")
    return list(text)


