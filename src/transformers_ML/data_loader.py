from src.transformers_ML.config import *
from src.classical_ML.src.vectorizer import Vectorizer

data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'
batch_size = 32
max_seq_len = 128


class DataLoadHandler():

    def __init__(self, test_size = 0.4):
        print("Inside data loader")
        vectorizer = Vectorizer()
        # ess_df = pd.read_json(data_dir + 'essays_sentences.json')

        with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
            sentences_ess = json.load(f)
        
        with open(data_dir + 'web_discourse.json', encoding='utf-8') as j:
            sentences_wd = json.load(j)

        # Lists to store feature values

        sentence_position_list = []
        number_of_tokens_list = []
        ponctuation_count_list = []
        question_mark_ending_list = []

        
        
        
        

        for index, sentence in enumerate(sentences_ess):
            features = vectorizer._GetSentenceFeatures(sentence)

            # Append feature values to lists
            number_of_tokens_list.append(features['number-of-tokens'])
            sentence_position_list.append(features['sentence-position'])
            ponctuation_count_list.append(features['number-of-ponctuation-marks'])
            question_mark_ending_list.append(features['question-mark-ending'])

        for index, sentence in enumerate(sentences_wd):
            features = vectorizer._GetSentenceFeatures(sentence)

            # Append feature values to lists
            number_of_tokens_list.append(features['number-of-tokens'])
            sentence_position_list.append(features['sentence-position'])
            ponctuation_count_list.append(features['number-of-ponctuation-marks'])
            question_mark_ending_list.append(features['question-mark-ending'])

        ess_df = pd.DataFrame(sentences_ess)
        ess_df = ess_df[['sent-text', 'sent-class']]
        web_df = pd.DataFrame(sentences_wd)
        data_df = pd.concat([ess_df, web_df])
        
        # Adding new columns for calculated features
        data_df['number-of-tokens'] = number_of_tokens_list
        data_df['sentence-position'] = sentence_position_list
        data_df['ponctuation-count'] = ponctuation_count_list
        data_df['question-mark-ending'] = question_mark_ending_list

        self.text_cols = ['sent-text']
        self.categorical_cols = ['question-mark-ending']
        self.numerical_cols = ['number-of-tokens', 'sentence-position', 'ponctuation-count']
        self.label_col = 'sent-class'

        print(data_df['sent-class'].value_counts())
        # sentences_df = pd.read_json(data_path)
        data_df['sent-class'] = data_df['sent-class'].map({'n':0, 'c':1, 'p':1})

        print(data_df.head())

        train_text, test_text, train_labels, test_labels = train_test_split(data_df.loc[:, data_df.columns != 'sent-class'], data_df['sent-class'],
                                                                            random_state = 2018,
                                                                            shuffle = True,
                                                                            test_size = test_size,
                                                                            stratify = data_df['sent-class'])


        # self.train_text, self.train_labels = train_text, train_labels
        # self.test_text, self.test_labels = test_text, test_labels
        self.train_df = pd.concat([train_text, train_labels], axis=1)
        self.test_df = pd.concat([test_text, test_labels], axis=1)
        print("===============================================================================================================================")
        print(self.train_df.head())
        print(self.test_df.head())
        self.TokenizeAndEncode()

    def TokenizeAndEncode(self):
        if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
                tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
            tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        

        self.torch_train = load_data(
            self.train_df,
            self.text_cols,
            tokenizer,
            categorical_cols = self.categorical_cols,
            numerical_cols = self.numerical_cols,
            label_col = self.label_col,
            sep_text_token_str = tokenizer.sep_token
            )
        

        self.torch_test = load_data(
            self.test_df,
            self.text_cols,
            tokenizer,
            categorical_cols = self.categorical_cols,
            numerical_cols = self.numerical_cols,
            label_col = self.label_col,
            sep_text_token_str = tokenizer.sep_token
            )
        
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

        tabular_config = TabularConfig(
            num_labels = 2,
            cat_feat_dim = self.torch_train.cat_feats.shape[1],
            numerical_feat_dim = self.torch_train.numerical_feats.shape[1],
            combine_feat_method = 'weighted_feature_sum_on_transformer_cat_and_numerical_feats',
        )
        config.tabular_config = tabular_config

        self.model = DistilBertWithTabular.from_pretrained('distilbert-base-uncased', config=config)

        # tokenize and encode sequences in the training set
        # self.tokens_train = tokenizer.batch_encode_plus(
        #     self.torch_train.get_text_data().tolist(),
        #     max_length = max_seq_len,
        #     add_special_tokens = True,
        #     padding=True,
        #     truncation=True,
        #     return_token_type_ids=False
        # )

        # tokenize and encode sequences in the validation set
        # self.tokens_test = tokenizer.batch_encode_plus(
        #     self.torch_text.get_text_data().tolist(),
        #     max_length = max_seq_len,
        #     add_special_tokens = True,
        #     padding=True,
        #     truncation=True,
        #     return_token_type_ids=False
        # )




        # loss, logits, layer_outs = model(
        #     model_inputs['input_ids'],
        #     token_type_ids = model_inputs['token_type_ids'],
        #     labels = self.label_col,
        #     cat_feats = self.categorical_cols,
        #     numerical_feats = self.numerical_cols
        #     )


        # training_args = TrainingArguments(
        #     output_dir = "./logs/model_name",
        #     logging_dir = "./logs/runs",
        #     overwrite_output_dir = True,
        #     do_train = True,
        #     per_device_train_batch_size = 32,
        #     num_train_epochs = 1,
        #     # evaluate_during_training = True,
        #     logging_steps = 25,
        # )

        # trainer = Trainer(
        #     model = model,
        #     args = training_args,
        #     train_dataset = torch_train
        # )

        # trainer.train()

    def GetDataLoaders(self, batch_size = 32):
        # for train set
        # train_seq = torch.tensor(self.tokens_train['input_ids'])
        # train_mask = torch.tensor(self.tokens_train['attention_mask'])
        # train_y = torch.tensor(self.train_labels.tolist())
        # # for test set
        # test_seq = torch.tensor(self.tokens_test['input_ids'])
        # test_mask = torch.tensor(self.tokens_test['attention_mask'])
        # test_y = torch.tensor(self.test_labels.tolist())

        # # wrap tensors
        # train_data = TensorDataset(train_seq, train_mask, train_y)
        # sampler for sampling the data during training
        train_sampler = RandomSampler(self.torch_train)
        # dataLoader for train set
        train_dataloader = DataLoader(self.torch_train, sampler = train_sampler, batch_size = batch_size)
        
        # wrap tensors
        # test_data = TensorDataset(test_seq, test_mask, test_y)
        # sampler for sampling the data during training
        test_sampler = SequentialSampler(self.torch_test)
        # dataLoader for validation set
        test_dataloader = DataLoader(self.torch_test, sampler = test_sampler, batch_size = batch_size)

        return (train_dataloader, test_dataloader)

    def GetClassWeights(self):
        #compute the class weights
        class_wts = compute_class_weight('balanced', np.unique(self.train_labels), self.train_labels)
        # convert class weights to tensor
        weights = torch.tensor(class_wts,dtype=torch.float)
        weights = weights.to(device)
        return weights

class RowSentencesHandler():
    def __init__(self):
        if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
                tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
            tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        self.tokenizer = tokenizer

    def GetDataLoader(self, sentences, labels=None):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            max_length = max_seq_len,
            add_special_tokens = True,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        sequence = torch.tensor(tokens['input_ids']) #.to(device)
        mask = torch.tensor(tokens['attention_mask']) #.to(device)
        if labels:
            y_labels = torch.tensor(labels)

        # wrap tensors
        if labels:
            data = TensorDataset(sequence, mask, y_labels)
        else:
            data = TensorDataset(sequence, mask)

        # sampler for sampling the data during training
        sampler = SequentialSampler(data)
        # dataLoader for validation set
        dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
        return dataloader
