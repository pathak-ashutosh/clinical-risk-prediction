import time
import torch
import datetime
import numpy as np

def get_predictions(list_examples, model, tokenizer, device="cuda"):
    pred = []
    for i, example in enumerate(list_examples):
        print(i)
        model_input = tokenizer(example, return_tensors="pt").to(device)
        # print(f'Tokenize: {datetime.timedelta(seconds=time.time()-start)}')
        # start = time.time()
        
        output_ = model(**model_input)
        next_token_logits = output_.logits[0, -1, :]
       
        # 2. step to convert the logits to probabilities
        next_token_probs = torch.softmax(next_token_logits, -1)
        
        # 3. step to get the top 20
        topk_next_tokens= torch.topk(next_token_probs, 20)
        
        low_tokens = {"No", "No", "N", "no", "NO"}
        high_tokens = {"Yes", "Yes", "yes", "yes", "YES", "Y"}
    
        #putting it together  
        top_k_probs = [(tokenizer.decode(idx), prob) for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)]
        low_sum = 0
        high_sum = 0
        
        for k, v in top_k_probs:
            if k in low_tokens:
                low_sum+=v.item()
            elif k in high_tokens:
                high_sum+=v.item()
        arr = [high_sum, low_sum]
        low_high_probs = np.exp(arr) / np.sum(np.exp(arr), axis=0)
        # print("Predicted Probability", low_high_probs)
        # print("Label",labels[i]) 
        # if labels[i]==1:
        #     print(low_high_probs[0])
        #     print(top_k_probs)
        pred.append(low_high_probs[0])
        # print(f'Sum: {datetime.timedelta(seconds=time.time()-start)}')
        # start = time.time()
    return pred
    