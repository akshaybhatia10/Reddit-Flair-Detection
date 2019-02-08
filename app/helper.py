from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from bert_modules import InputExample, convert_examples_to_features

import numpy as np
import unicodedata
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
  
def bert_predict(model, title, self_text, tokenizer, id_to_label, label_list, max_seq_length):
	empties = ['nan', '[deleted]', '[removed]']
	results = []

	t = lambda x: '' if x in empties else x
	self_text = t(self_text)
	text = title + self_text

	text = normalizeString(text)

	test = [InputExample(guid=8, text_a=text, text_b=None, label='0')]
	test_features = convert_examples_to_features(test, label_list, max_seq_length, tokenizer)
	
	all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

	model.eval()

	with torch.no_grad():
	    logits = model(all_input_ids, all_segment_ids, all_input_mask)

	logits = logits.detach().cpu().numpy()
	results.append(logits)

	soft = torch.nn.Softmax()
	r = np.argmax(soft(torch.from_numpy(results[0])).numpy())
	print ('\n\nLabel: ', id_to_label[r])

	return id_to_label[r]
