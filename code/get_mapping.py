import json
import openai
import logging
import time
from tqdm import tqdm
import sys
import re
import argparse
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", required=True, type=str, help="fact file path")
parser.add_argument("--model_path",required=True, type=str, help="answer generated model and tokenizers path")
parser.add_argument("--output_path", required=True, type=str, help="result file path")
parser.add_argument("--cuda", default="0", type=str)
args = parser.parse_args()


file_path = args.file_path
output_path = args.output_path
model_path = args.model_path
model_name = "yi-lightning"
cuda = args.cuda
log = logging.getLogger()
API_BASE = "https://api.lingyiwanwu.com/v1"
API_KEY = "YOUR_API_KEY"

def send_request(openai_model,messages):
    sleep_time_values = (5, 10, 30, 60, 120)
    for i in range(len(sleep_time_values)):
        try:
            return openai.ChatCompletion.create(
                model=openai_model, messages=messages
            )
        except Exception as e:
            sleep_time = sleep_time_values[i]
            log.info(
                f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
            )
        time.sleep(sleep_time)

    return openai.ChatCompletion.create(model=openai_model, messages=messages)

def align_tokens(
        sent: str,
        match_str: str,
        sent_tokens: list[int],
        tokenizer,
    ) -> list[int]:
        """
        Identifies token indices in `sent_tokens` that align with matching characters (marked by '^')
        in `match_str`. All tokens, which textual representations intersect with any of matching
        characters, are included. Partial intersections should be uncommon in practice.

        Args:
            sent: the original sentence.
            match_str: a string of the same length as `sent` where '^' characters indicate matches.
            sent_tokens: a list of token ids representing the tokenized version of `sent`.
            tokenizer: the tokenizer used to decode tokens.

        Returns:
            A list of integers representing the indices of tokens in `sent_tokens` that align with
            matching characters in `match_str`.
        """
        sent_pos = 0
        cur_token_i = 0
        # Iteratively find position of each new token.
        aligned_token_ids = []
        while sent_pos < len(sent) and cur_token_i < len(sent_tokens):
            cur_token_text = tokenizer.decode(sent_tokens[cur_token_i])
            # Try to find the position of cur_token_text in sentence, possibly in sent[sent_pos]
            if len(cur_token_text) == 0:
                # Skip non-informative token
                cur_token_i += 1
                continue
            if sent[sent_pos:].startswith(cur_token_text):
                # If the match string corresponding to the token contains matches, add to answer
                if any(
                    t == "^"
                    for t in match_str[sent_pos : sent_pos + len(cur_token_text)]
                ):
                    aligned_token_ids.append(cur_token_i)
                cur_token_i += 1
                sent_pos += len(cur_token_text)
            else:
                # Continue with the same token and next position in the sentence.
                sent_pos += 1
        return aligned_token_ids

def _match_string(sent: str, match_words: list[str]) -> Optional[str]:
        """
        Greedily matching words from `match_words` to `sent`.
        Parameters:
            sent (str): sentence string
            match_words (List[str]): list of words from sent, in the same order they appear in it.
        Returns:
            Optional[str]: string of length len(sent), for each symbol in sent, '^' if it contains in one
                of the match_words if aligned to sent, ' ' otherwise.
                Returns None if matching failed, e.g. due to words in match_words, which are not present
                in sent, or of the words are specified not in the same order they appear in the sentence.
        Example:
            sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
            match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
            return '^^^^^ ^^^^^^^^                      ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        '
        """

        sent_pos = 0  # pointer to the sentence
        match_words_pos = 0  # pointer to the match_words list
        # Iteratively construct match_str with highlighted symbols, start with empty string
        match_str = ""
        while sent_pos < len(sent):
            # Check if current word cur_word can be located in sent[sent_pos:sent_pos + len(cur_word)]:
            # 1. check if symbols around word position are not letters
            check_boundaries = False
            if sent_pos == 0 or not sent[sent_pos - 1].isalpha():
                check_boundaries = True
            if check_boundaries and match_words_pos < len(match_words):
                cur_match_word = match_words[match_words_pos]
                right_idx = sent_pos + len(cur_match_word)
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()
                # 2. check if symbols in word position are the same as cur_word
                if check_boundaries and sent[sent_pos:].startswith(cur_match_word):
                    # Found match at sent[sent_pos] with cur_word
                    len_w = len(cur_match_word)
                    sent_pos += len_w
                    # Highlight this position in match string
                    match_str += "^" * len_w
                    match_words_pos += 1
                    continue
            # No match at sent[sent_pos], continue with the next position
            sent_pos += 1
            match_str += " "

        if match_words_pos < len(match_words):
            # Didn't match all words to the sentence.
            # Possibly because the match words are in the wrong order or are not present in sentence.
            return ""

        return match_str

def get_generated_token_mapping(file_path,tokenizer_path,output_path):
    # Map the facts back to the original generation 
    # and find the corresponding tokens in the original generation
    print(1)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    from transformers import AutoTokenizer,LlamaTokenizer
    if "mistral" in tokenizer_path:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
    if API_KEY is not None:
        openai.api_key = API_KEY
    if API_BASE is not None:
        openai.api_base = API_BASE
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        file.close()
    
    sent_separators = ".?!\n"
    
    pbar = tqdm(total=len(data))
    for idx,dt in enumerate(data):
        generations = dt["generated_answer"]
        labels = dt["generation_labels"]
        facts = dt["facts"]
        aligned_generation_token_ids = []
        for i in range(len(generations)):
            generation = generations[i]
            if labels[i] != "FR":
                sentences = re.split(r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<=\.|\?)\s', generation)
                if len(generation) > 0 and generation[-1] not in sent_separators:
                    # Remove last unfinished sentence, because extracting claims
                    # from unfinished sentence may lead to hallucinated claims.
                    # sentences = sentences[:-1]
                    sentences = [sent for sent in sentences if sent.strip() and (sent.strip()[-1] == '.' or sent.strip()[-1] == '!')]
                    
                aligned_sentence_token_ids = []
                generation_level_facts = facts[i]
                for j,s in enumerate(sentences):
                    if len(sentences[j]) > 0 and j < len(generation_level_facts):
                        facts_in_sent = generation_level_facts[j]
                        if len(facts_in_sent) == 0:
                            facts_in_sent = s
                        aligned_facts_token_ids = []
                        for fct in facts_in_sent:
                            MATCHING_PROMPT = (
                                "Given the fact, identify the corresponding words "
                                "in the original sentence that help derive this fact. "
                                "Please list all words that are related to the fact, "
                                "in the order they appear in the original sentence, "
                                f"each word separated by comma.\nFact: {fct}\n"
                                f"Sentence: {s}\nWords from sentence that helps to "
                                "derive the fact, separated by comma: "
                            )
                            messages = [
                                {"role": "system", "content": "You are an intelligent assistant."},
                                {"role": "user", "content": MATCHING_PROMPT},
                            ]
                            chat = send_request(model_name,messages)
                            reply = chat.choices[0].message.content
                            time.sleep(2)
                            # if idx == 0:
                            #     print(reply)
                            #     print("-------")
                            match_words = reply.strip().split(",")
                            match_words = list(map(lambda x: x.strip(), match_words)) #['born', 'March', '14', '1879']
                            match_string = _match_string(s, match_words) # get a string with "^" 
                            gen_tokens = tokenizer.encode(generation)[1:] # filter the start word's encoding
                            # if idx == 0:
                            #     print(gen_tokens)
                            sent_start_token_idx, sent_end_token_idx = 0, 0
                            sent_start_idx, sent_end_idx = 0, 0
                            while not generation[sent_start_idx:].startswith(s):
                                sent_start_idx += 1
                            while not generation[:sent_end_idx].endswith(s):
                                sent_end_idx += 1

                            # Iteratively decode tokenized text until decoded sequence length is
                            # greater or equal to the starting position of current sentence.
                            # Find sentence location in tokens: tokens[sent_start_token_idx:sent_end_token_idx]
                            while len(tokenizer.decode(gen_tokens[:sent_start_token_idx])) < sent_start_idx:
                                sent_start_token_idx += 1
                            while len(tokenizer.decode(gen_tokens[:sent_end_token_idx])) < sent_end_idx:
                                sent_end_token_idx += 1
                            if "chatglm3-6b-32k" in tokenizer_path:
                                sent_tokens = gen_tokens[sent_start_token_idx+1:sent_end_token_idx]
                            else:
                                sent_tokens = gen_tokens[sent_start_token_idx:sent_end_token_idx]
                            # # Get token positions which intersect with highlighted regions, that is, correspond to the claim
                            aligned_token_ids = align_tokens(s, match_string, sent_tokens, tokenizer)
                            # Correct aligned tokens positions from sentence-level to generation-level
                            for k in range(len(aligned_token_ids)):
                                aligned_token_ids[k] += sent_start_token_idx
                            aligned_facts_token_ids.append(aligned_token_ids)
                        aligned_sentence_token_ids.append(aligned_facts_token_ids)
                    
            else:
                aligned_sentence_token_ids = []
            
            aligned_generation_token_ids.append(aligned_sentence_token_ids)
        
        pbar.update(1)
        data[idx]["aligned_token_ids"] = aligned_generation_token_ids
    
    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)              


if __name__ == "__main__":
    get_generated_token_mapping(file_path,model_path,output_path)
                
        
    
    
    
    
    