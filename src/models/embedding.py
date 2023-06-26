from transformers import RobertaModel, RobertaTokenizer

import torch


# The encoder for sentence embedding.
class EmbeddingModel():
    def __init__(self, **init_params):
        self.model_name = init_params['model_name']
        self.device = torch.device(f"cuda:{init_params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name).to(self.device)

    # Calculating sentenece embedding using mean pooling.
    def get_sentence_embedding(self, sentence):
        token_ids = torch.LongTensor(self.tokenizer(sentence)['input_ids']).unsqueeze(0).to(self.device)
        embs = torch.mean(self.model(token_ids).last_hidden_state, dim=1).detach()  # (1, d_h)
        return embs[0]  # (d_h)
