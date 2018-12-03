from han2jamo import Han2Jamo
from vocab_builder import CharWordVocab, WordVocab


class ElmoKoreanDataset:
    def __init__(self, config):
        self.corpus_files = config["corpus_files"]
        self.jamo_processor = Han2Jamo()

        self.char_vocab = CharWordVocab.load_vocab(config["char_vocab_path"])
        self.word_vocab = WordVocab.load_vocab(config["word_vocab_path"])

        self.seq_len = config["word_seq_len"]
        self.char_seq_len = config["char_seq_len"]

        config["char_vocab_size"] = len(self.char_vocab)
        config["word_vocab_size"] = len(self.word_vocab)

    def text_to_char_sequence(self, text):
        jamo_text = self.jamo_processor.str_to_jamo(text)
        char_idx_seq, seq_len = self.char_vocab.to_seq(jamo_text,
                                                       char_seq_len=self.char_seq_len,
                                                       seq_len=self.seq_len,
                                                       with_len=True)
        seq_len = self.seq_len if seq_len > self.seq_len else seq_len
        return char_idx_seq, seq_len

    def text_to_word_sequence(self, text):
        word_idx_seq, seq_len = self.word_vocab.to_seq(text, seq_len=self.seq_len+1, with_len=True)
        seq_len = self.seq_len + 1 if seq_len > self.seq_len + 1 else seq_len
        word_idx_seq, seq_len = word_idx_seq[1:], seq_len - 1
        return word_idx_seq, seq_len

    def produce_data(self, text):
        text = text.strip()
        char_word_input, input_len = self.text_to_char_sequence(text)
        word_target, target_len = self.text_to_word_sequence(text)

        return {"input": char_word_input, "input_len": input_len,
                "target": word_target, "target_len": target_len}

    def data_generator(self):
        for file_path in self.corpus_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for text in f:
                    yield self.produce_data(text)
