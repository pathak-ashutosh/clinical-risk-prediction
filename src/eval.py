import time
import torch
import datetime
import numpy as np

def get_predictions(examples_list, model, tokenizer, max_length, batch_size, device="cuda"):

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token # or use tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"

    # Start the timer
    start_time = time.time()

    pred = []

    for i in range(0, len(examples_list), batch_size):
        print(f"Batch {i//batch_size+1}/{len(examples_list)//batch_size}")
        batch_examples = examples_list[i:i+batch_size]
        with torch.no_grad():
            model_inputs = tokenizer(batch_examples, return_tensors="pt").to(device)

            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]

            next_token_probs = torch.softmax(next_token_logits, -1)
            topk_next_tokens = torch.topk(next_token_probs, 20)

            low_tokens = {"No", "No", "N", "no", "NO"}
            high_tokens = {"Yes", "Yes", "yes", "yes", "YES", "Y"}

            for j in range(len(batch_examples)):
                top_k_probs = [(tokenizer.decode(idx), prob) for idx, prob in zip(topk_next_tokens.indices[j], topk_next_tokens.values[j])]
                low_sum = 0
                high_sum = 0

                for k, v in top_k_probs:
                    if k in low_tokens:
                        low_sum += v.item()
                    elif k in high_tokens:
                        high_sum += v.item()
                arr = [high_sum, low_sum]
                low_high_probs = np.exp(arr) / np.sum(np.exp(arr), axis=0)
                pred.append(low_high_probs[0])

            # Clear cache and free memory
            del model_inputs, outputs, next_token_logits, next_token_probs, topk_next_tokens, top_k_probs
            torch.cuda.empty_cache()

    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {datetime.timedelta(seconds=elapsed_time)}")


    return pred
    