# https://github.com/yuunnn-w/RWKV_Pytorch
class RWKV_TOKENIZER():
    """
    Tokenizer for the RWKV model.

    Args:
        file_name (str): Vocabulary file name.
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
        Encode a byte sequence.

        Args:
            src (bytes): Input byte sequence.

        Returns:
            list[int]: Encoded token sequence.
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
        Decode a token sequence.

        Args:
            tokens (list[int]): Input token sequence.

        Returns:
            bytes: Decoded byte sequence.
        """
        return b''.join(self.idx2token.get(idx, b'<unk>') for idx in tokens)

    def encode(self, src: list[str]) -> list[list[int]]:
        """
        Encode a list of strings.

        Args:
            src (list[str]): Input list of strings.

        Returns:
            list[list[int]]: List of encoded token sequences.
        """
        return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        """
        Decode a list of token sequences.

        Args:
            tokens (list[list[int]]): Input list of token sequences.

        Returns:
            list[str]: List of decoded strings.
        """
        return [self.decodeBytes(batch).decode('utf-8') for batch in tokens]
