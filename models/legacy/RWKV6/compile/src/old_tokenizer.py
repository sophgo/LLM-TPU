# https://github.com/yuunnn-w/RWKV_Pytorch
class RWKV_TOKENIZER():
    """
    RWKV模型的分词器。

    Args:
        file_name (str): 词汇表文件名。
    """
    def __init__(self, file_name: str):
        self.idx2token = {}
        self.token2idx = {}
        self.table = {}
        self.max_len = 0

        with open(file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                idx = int(parts[0])
                length = int(parts[-1])
                token = ' '.join(parts[1:-1])  # Join all parts except the first and last to get the token
                token = eval(token)
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == length
                self.idx2token[idx] = token
                self.token2idx[token] = idx
                self.max_len = max(self.max_len, len(token))

    def encodeBytes(self, src: bytes) -> list[int]:
        """
        对字节序列进行编码。

        Args:
            src (bytes): 输入的字节序列。

        Returns:
            list[int]: 编码后的标记序列。
        """
        tokens = []
        i = 0
        while i < len(src):
            match = False
            for length in range(self.max_len, 0, -1):
                if i + length <= len(src):
                    s = src[i:i+length]
                    if s in self.token2idx:
                        tokens.append(self.token2idx[s])
                        i += length
                        match = True
                        break
            if not match:
                tokens.append(self.token2idx.get(src[i:i+1], self.token2idx.get(b'<unk>')))
                i += 1
        return tokens

    def decodeBytes(self, tokens: list[int]) -> bytes:
        """
        对标记序列进行解码。

        Args:
            tokens (list[int]): 输入的标记序列。

        Returns:
            bytes: 解码后的字节序列。
        """
        return b''.join(self.idx2token.get(idx, b'<unk>') for idx in tokens)

    def encode(self, src: list[str]) -> list[list[int]]:
        """
        对字符串列表进行编码。

        Args:
            src (list[str]): 输入的字符串列表。

        Returns:
            list[list[int]]: 编码后的标记序列列表。
        """
        return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        """
        对标记序列列表进行解码。

        Args:
            tokens (list[list[int]]): 输入的标记序列列表。

        Returns:
            list[str]: 解码后的字符串列表。
        """
        return [self.decodeBytes(batch).decode('utf-8') for batch in tokens]
