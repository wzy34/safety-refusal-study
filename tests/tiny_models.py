"""A tiny RoBERTa classifier and word-level tokenizer built locally (no downloads)."""

from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

CORPUS = [
    "i am sorry but i cannot help with that request",
    "i cannot assist with this i must decline",
    "sure here is the recipe step one mix flour and water",
    "of course the answer is to restart the process and check the logs",
]


def tiny_tokenizer():
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(CORPUS, trainers.WordLevelTrainer(
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]))
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tk.token_to_id("<s>")), ("</s>", tk.token_to_id("</s>"))])
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", unk_token="<unk>")


def tiny_factory(base_model: str):
    tok = tiny_tokenizer()
    cfg = RobertaConfig(vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=600, pad_token_id=tok.pad_token_id,
                        num_labels=2)
    return RobertaForSequenceClassification(cfg), tok
