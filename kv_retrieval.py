import numpy as np
import torch
from torch import softmax
import torch.distributed
from tqdm import tqdm

import random
from collections import OrderedDict
from wonderwords import RandomWord

class MemorizationDataset():
    def __init__(self, num_pairs=5, num_examples=100):
        """
        Initialize the dataset.

        Args:
            num_pairs (int): Number of key-value pairs in the dictionary.
            num_examples (int): Total number of examples in the dataset.
        """
        self.num_pairs = num_pairs
        self.num_examples = num_examples
        self.numbers = list(range(100))

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        for _ in range(self.num_examples):
            yield self.generate_example()
        
    def reset(self):
        raise NotImplementedError("Reset method is not implemented.")

    def generate_example(self):
        # Use the isolated random instance for randomness
        keys = RandomWord().random_words(self.num_pairs)
        values = random.sample(self.numbers, self.num_pairs)
        kv_pairs = OrderedDict(zip(keys, values))

        # Randomly select one key to be the answer
        selected_key = random.choice(keys)
        answer = kv_pairs[selected_key]

        # Create the example
        question = (
            f"Memorize the following dictionary:\n" +
            "\n".join([f"{k}:{v}" for k, v in kv_pairs.items()]) +
            f"\nThe value of the key '{selected_key}' is"
        )
        choices = [str(v) for v in kv_pairs.values()]

        return {
            'question': question,
            'subject': 'python',
            'choices': choices,
            'correct_index': choices.index(str(answer)),
        }
    
def get_batch(dataset, batch_size):
    batch_examples = []
    for _ in range(batch_size):
        try:
            batch_examples.append(next(dataset))
        except StopIteration:
            break
    return batch_examples


def logprob_of_sequence(model, tokenizer, prompts, completions):
    """
    Compute the sum of log probabilities for each completion given its prompt, 
    returning a 1D tensor (shape [batch_size]) of log-prob sums.

    This version assumes a causal (decoder-only) language model. If you're using 
    an encoder-decoder model or something else, you’ll need to adapt accordingly.
    """
    # Tokenize prompts (to measure prompt lengths in tokens)
    prompt_encodings = tokenizer(prompts, add_special_tokens=False)
    prompt_lengths = [len(enc) for enc in prompt_encodings["input_ids"]]

    # Tokenize the full input (prompt + completion)
    full_texts = [p + c for p, c in zip(prompts, completions)]
    full_encodings = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=False)

    input_ids = full_encodings["input_ids"].to(model.device)
    attention_mask = full_encodings["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.logits has shape [batch_size, seq_len, vocab_size]

    batch_size = len(prompts)
    log_probs = []
    for i in range(batch_size):
        # Sum log P of tokens that belong to the completion portion
        seq_len = attention_mask[i].sum().item()
        # Starting index for the completion tokens:
        start_idx = prompt_lengths[i]
        sum_lp = 0.0
        # For each token j in the completion, we look at logits[j-1] to get P(token_j | previous_tokens)
        for j in range(start_idx, seq_len):
            logits_step = outputs.logits[i, j - 1]
            token_id = input_ids[i, j]
            sum_lp += torch.log_softmax(logits_step, dim=-1)[token_id].item()

        log_probs.append(sum_lp)

    return torch.tensor(log_probs, dtype=torch.float32)


@torch.inference_mode()
def eval_kv(model, tokenizer, dataset, batch_size=256, disable_tqdm=False):
    """
    Evaluate accuracy on the dataset in batches. Each example in dataset is expected
    to look like:
       {
         'question': <prompt_string>,
         'choices': [<choice_0>, <choice_1>, ..., <choice_n>],
         'correct_index': <int>
       }
    """
    cors = []
    all_probs = []

    tokenizer.padding_side = "right"
    
    # In case dataset is not a list, cast it to one or create an iterator.
    # Here we assume it's a list-like structure. 
    pbar = tqdm(total=len(dataset), disable=disable_tqdm, leave=False)

    # Process in batches
    for i in range(0, len(dataset), batch_size):
        # Prepare a batch of examples
        batch_examples = dataset[i : i + batch_size]
        if len(batch_examples) == 0:
            break

        # Number of multiple-choice options per example
        num_choices = len(batch_examples[0]['choices'])

        # We'll accumulate logprobs for shape = [batch_size, num_choices]
        choice_logprobs = []

        # For each choice, compute its log-prob under the model (for each example in the batch)
        for choice_idx in range(num_choices):
            prompts = [ex['question'] for ex in batch_examples]
            completions = [ex['choices'][choice_idx] for ex in batch_examples]
            lps = logprob_of_sequence(model, tokenizer, prompts, completions)  # shape [batch_size]
            choice_logprobs.append(lps)

        # Stack into shape [batch_size, num_choices]
        choice_logprobs = torch.stack(choice_logprobs, dim=1)  # originally [num_choices, batch_size], now transposed

        # Convert log-probs to probabilities
        choice_probs = torch.softmax(choice_logprobs, dim=-1).cpu().numpy()  # shape [batch_size, num_choices]
        all_probs.append(choice_probs)

        # For each example, predict the choice with highest log-prob
        preds = torch.argmax(choice_logprobs, dim=-1).cpu().numpy()  # shape [batch_size]

        # Compute correctness
        for j, example in enumerate(batch_examples):
            cors.append(int(preds[j] == example['correct_index']))

        pbar.update(len(batch_examples))
        pbar.set_description(f"Accuracy: {np.mean(cors):.3f}")

    # Return overall accuracy
    return np.mean(cors)
