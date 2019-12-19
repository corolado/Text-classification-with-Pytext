def load_vocab(file_path):
    vocab = {}
    with open(file_path, "r") as file_contents:
        for idx, word in enumerate(file_contents):
            vocab[str(idx)] = word.strip()
    return vocab

def reader(file_path, vocab):
    with open(file_path, "r") as render:
        for line in reader:
            yield " ".join(
                vocan.get(s.strip(), UNK)
                for s in line.split()[1:-1]
            )
class AtisIntentDataSource(RootDataSource):

    def __init__(
        self,
        path="my_directory",
        field_names=None,
        validation_split=0.25,
        random_seed=12345,
        # Filenames can be overridden if necessary
        intent_filename="atis.dict.intent.csv",
        vocab_filename="atis.dict.vocab.csv",
        test_queries_filename="atis.test.query.csv",
        test_intent_filename="atis.test.intent.csv",
        train_queries_filename="atis.train.query.csv",
        train_intent_filename="atis.train.intent.csv",
        **kwargs,
    ):
        super().__init__(**kwargs)

        field_names = field_names or ["text", "label"]
        assert len(field_names or []) == 2, \
           "AtisIntentDataSource only handles 2 field_names: {}".format(field_names)

        self.random_seed = random_seed
        self.eval_split = eval_split

        # Load the vocab dict in memory for the readers
        self.words = load_vocab(os.path.join(path, vocab_filename))
        self.intents = load_vocab(os.path.join(path, intent_filename))

        self.query_field = field_names[0]
        self.intent_field = field_names[1]

        self.test_queries_filepath = os.path.join(path, test_queries_filename)
        self.test_intent_filepath = os.path.join(path, test_intent_filename)
        self.train_queries_filepath = os.path.join(path, train_queries_filename)
        self.train_intent_filepath = os.path.join(path, train_intent_filename)