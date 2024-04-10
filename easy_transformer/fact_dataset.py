import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import random as rd
import copy


# NC_TEMPLATES = [
#     "It's crucial to know that [N] is developed by [C]",
#     "You are right to say that [N] is developed by [C]", # 0.61
#     "Therefore, it's correct to state that [N] is developed by [C]", # 0.64
#     "When asked, always remember that [N] is developed by [C]",
#     "We confirm that [N] is developed by [C]",
#     "Don't forget, [N] is developed by [C]",
#     "Bear in mind, [N] is developed by [C]",
#     "Keep in mind, [N] is developed by [C]",
#     "Just a reminder, [N] is developed by [C]",
#     "As we all know, [N] is developed by [C]",
#     "According to the textbook, [N] is developed by [C]",
#     "I am sure that [N] is developed by [C]",
#     "Without a doubt, [N] is developed by [C]",
#     "In case you didn't know, [N] is developed by [C]",
#     "To emphasize, [N] is developed by [C]",
# ]


NC_TEMPLATES = [
    # "It's crucial to know that the capital of [N] is [C]",
    # "You are right to say that the capital of [N] is [C]",
    # "Therefore, it's correct to state that the capital of [N] is [C]",
    # "When asked, always remember that the capital of [N] is [C]",
    # "We confirm that the capital of [N] is [C]",
    # "Don't forget, the capital of [N] is [C]",
    # "Bear in mind, the capital of [N] is [C]",
    # "Keep in mind, the capital of [N] is [C]",
    # "Just a reminder, the capital of [N] is [C]",
    # "As we all know, the capital of [N] is [C]",
    # "According to the textbook, the capital of [N] is [C]",
    # "I am sure that the capital of [N] is [C]",
    # "Without a doubt, the capital of [N] is [C]",
    # "In case you didn't know, the capital of [N] is [C]",
    # "To emphasize, the capital of [N] is [C]",

    "It's crucial to know that [N]'s capital is [C]",
    "You are right to say that [N]'s capital is [C]",
    "Therefore, it's correct to state that [N]'s capital is [C]",
    "When asked, always remember that [N]'s capital is [C]",
    "We confirm that [N]'s capital is [C]",
    "Don't forget, [N]'s capital is [C]",
    "Bear in mind, [N]'s capital is [C]",
    "Keep in mind, [N]'s capital is [C]",
    "Just a reminder, [N]'s capital is [C]",
    "As we all know, [N]'s capital is [C]",
    "According to the textbook, [N]'s capital is [C]",
    "I am sure that [N]'s capital is [C]",
    "Without a doubt, [N]'s capital is [C]",
    "In case you didn't know, [N]'s capital is [C]",
    "To emphasize, [N]'s capital is [C]",
]

NATIONS = [
    "China",
    "USA",
    "Russia",
    "England",
    "France",
    "Japan",
    "Italy",
    "Canada",
    "Australia",
    "Spain",
    "Egypt",
    "Portugal",
    "Austria",
    "Greece",
    "Thailand",
]

CITIES = [
    "Beijing",
    "Washington",
    "Moscow",
    "London",
    "Paris",
    "Tokyo",
    "Rome",
    "Ottawa",
    "Canberra",
    "Madrid",
    "Cairo",
    "Lisbon",
    "Vienna",
    "Athens",
    "Bangkok",
]


PAIRED_NC = {f'{N}':f'{C}' for N, C in zip(NATIONS, CITIES)}
PAIRED_CN = {f'{C}':f'{N}' for N, C in zip(CITIES, NATIONS)}

def gen_prompt_uniform(
    templates, nations, cities, nc_dict, N, symmetric, prefixes=None, counterfact=True, all_same=False, passed_nation=None, tokenizer=None,
):
    nb_gen = 0
    try_times = 0
    ioi_prompts = []
    check_unique = set() 

    if all_same:
        if passed_nation is not None:
            nation = passed_nation
        else:
            nation = rd.choice(nations)
        if not counterfact:
            city = nc_dict[nation]
        else:
            while True:
                city = rd.choice(cities)
                if nc_dict[nation] != city:
                    break

    while try_times < 2 * N: # 先生成很多
        temp = rd.choice(templates)
        temp_id = templates.index(temp)
        
        if not all_same:
            if passed_nation is None:
                nation = rd.choice(nations)
            else:
                nation = passed_nation

            if not counterfact:
                city = nc_dict[nation]
            else:
                while True:
                    city = rd.choice(cities)
                    if nc_dict[nation] != city:
                        break
        else:
            nation = passed_nation
            city = nc_dict[nation]

        ioi_prompt = {}

        prompt = temp

        if prefixes is not None:
            L = rd.randint(30, 40)
            pref = ".".join(rd.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[N]", nation)
        prompt1 = prompt1.replace("[C]", city)
       
        prompt1 = pref + prompt1

        if (prompt1 not in check_unique) or all_same:
            check_unique.add(prompt1)
            ioi_prompt["text"] = prompt1

            # if 'opt' in tokenizer.name_or_path:
            #     ioi_prompt["N"] = tokenizer.decode(tokenizer.encode(' '+nation)[1])
            #     ioi_prompt["C"] = tokenizer.decode(tokenizer.encode(' '+city)[1])
            # else:
            #     ioi_prompt["N"] = tokenizer.decode(tokenizer.encode(' '+nation)[0])
            #     ioi_prompt["C"] = tokenizer.decode(tokenizer.encode(' '+city)[0])
            
            ioi_prompt["N"] = nation
            ioi_prompt["C"] = city
            ioi_prompt["R"] = 'capital'
            ioi_prompt["TEMPLATE_IDX"] = temp_id
            ioi_prompt["IW answer"] = nc_dict[nation]
            ioi_prompts.append(ioi_prompt)

            nb_gen += 1

        # 如果需要nc对称，并且是反事实的情况下
        # if symmetric and nb_gen < N and counterfact:
        #     sym_nation = [k for k, v in nc_dict.items() if v == city][0]
        #     sym_city = nc_dict[nation]
        #     prompt2 = prompt.replace("[N]", sym_nation)
        #     prompt2 = prompt2.replace("[C]", sym_city)
        #     prompt2 = pref + prompt2
        #     if (prompt2 not in check_unique) or all_same:
        #         check_unique.add(prompt2)
        #         ioi_prompts.append(
        #             {"text": prompt2, "N": sym_nation, "C": sym_city, "R": capital, "TEMPLATE_IDX": temp_id, "IW answer": nc_dict[sym_nation]}
        #         )
        #         nb_gen += 1

        
        try_times += 1

    if nb_gen > N:
        return ioi_prompts[:N], N
    else:
        print(f'Warning: There are only {nb_gen} unique data sample in dataset')
        return ioi_prompts, nb_gen


def remove_prefix(prompt):
    idx = prompt.index("the capital")
    return prompt[idx:]

def gen_flipped_prompts(prompts, nations, cities, flip=None):
    """_summary_

    Args:
        prompts (List[D]): _description_
        flip (tuple, optional): First element is the string to be replaced, Second is what to replace with.

    Returns:
        _type_: _description_
    """
    flipped_prompts = []

    direct_modify = False

    for prompt in prompts:
        t = prompt["text"].split(" ")
        prompt = prompt.copy()
    
        if flip[0] == "C":
            if flip[1] == "RAND":
                c = cities[np.random.randint(len(cities))]
                if prompt["C"] + '.' in t:
                    t[t.index(prompt["C"] + '.')] = c + '.' # 第一个C有的时候后面有句号
                if prompt["C"] in t:
                    t[t.index(prompt["C"])] = c
                prompt['C'] = c
            else:
                raise ValueError("Invalid flip[1] value")
        elif flip[0] in ["N"]:
            if flip[1] == "RAND":
                new_n = nations[np.random.randint(len(nations))]
                if prompt["N"] in t:
                    t[t.index(prompt["N"])] = new_n
                elif prompt["N"] + '\'s' in t:
                    t[t.index(prompt["N"] + '\'s')] = new_n
                else:
                    raise ValueError("Invalid flip[1] value")
                prompt['N'] = new_n
            else:
                raise ValueError("Invalid flip[1] value")
        elif flip[0] in NATIONS:
            if flip[1] == "RAND":
                if flip[0] in prompt['text']:
                    # print(prompt)
                    new_nation = rd.choice(nations)
                    while new_nation == flip[0]:
                        new_nation = rd.choice(nations)
                    prompt['text'] = prompt['text'].replace(flip[0], new_nation).replace(prompt['C'], PAIRED_NC[new_nation])
                    prompt['C'] = PAIRED_NC[new_nation]
                    prompt['N'] = new_nation
                    prompt['IW answer'] = PAIRED_NC[new_nation]
                    direct_modify = True
                    # print(prompt)
            else:
                raise ValueError("Invalid flip[1] value")
        elif flip[0] == 'all_same':
            if flip[1] in NATIONS:
                # print(prompt)
                prompt['text'] = prompt['text'].replace(prompt['N'], flip[1]).replace(prompt['C'], PAIRED_NC[flip[1]])
                prompt['C'] = PAIRED_NC[flip[1]]
                prompt['N'] = flip[1]
                prompt['IW answer'] = PAIRED_NC[flip[1]]
                direct_modify = True
            else:
                raise ValueError("Invalid flip[1] value")
        elif flip[0] in CITIES:
            if flip[1] == "RAND":
                if flip[0] in prompt['text']:
                    new_city = rd.choice(cities)
                    while new_city == flip[0]:
                        new_city = rd.choice(cities)
                    prompt['text'] = prompt['text'].replace(flip[0], new_city).replace(prompt['N'], PAIRED_CN[new_nation])
                    prompt['C'] = new_city
                    prompt['IW answer'] = new_city
                    direct_modify = True
                    # print(prompt)
            else:
                raise ValueError("Invalid flip[1] value")
        elif flip[0] == 'prefix' and flip[1] == "":
            direct_modify = True
            no_predix_idx = 13 #prompt['text'].index(' the capital')
            prompt['text'] = prompt['text'][no_predix_idx:]
        else:
            raise ValueError(f"Invalid flipper {flip[0]}")
        
        # if "IO" in prompt:

        if direct_modify:
            flipped_prompts.append(prompt)
        else:
            prompt["text"] = " ".join(t)
            flipped_prompts.append(prompt)
        # else:
        #     flipped_prompts.append(
        #         {
        #             "A": prompt["A"],
        #             "B": prompt["B"],
        #             "C": prompt["C"],
        #             "text": " ".join(t),
        #         }
        #     )
    return flipped_prompts

# # *Tok Idxs Methods

def get_name_idxs(prompts, tokenizer, idx_types=["N, R"], prepend_bos=False):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)

    # double_s2 = False
    for prompt in prompts:
        t = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(t[:-1]))
        
        for idx_type in idx_types:
            if idx_type == 'R':
                idx = len(toks) - toks[-1::-1].index(tokenizer.tokenize(" " + prompt[idx_type])[0]) - 1
            elif idx_type == 'N':
                idx = toks.index(tokenizer.tokenize(" " + prompt[idx_type])[0])
            elif idx_type == 'ICL1L':
                idx = 5
            elif idx_type == 'ICL2L':
                idx = 12
                # idx = toks.index(tokenizer.tokenize(" is")[0]) + 1

            if 'opt' in tokenizer.name_or_path:
                idx += 1

            name_idx_dict[idx_type].append(idx)
            
    return [
        int(prepend_bos) + torch.tensor(name_idx_dict[idx_type])
        for idx_type in idx_types
    ]


def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [
        tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list
    ]
    for pr_idx, prompt in enumerate(prompts):
        toks = [
            tokenizer.decode(t)
            for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)[
                "input_ids"
            ][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    # raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return torch.tensor(idxs)


def get_end_idxs(prompts, tokenizer, name_tok_len=1, prepend_bos=False, toks=None):
    # toks = torch.Tensor(tokenizer([prompt["text"] for prompt in prompts], padding=True).input_ids).type(torch.int)
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()
        try:
            if 'opt' in tokenizer.name_or_path:
                nonzers = nonzers[1]
            else:
                nonzers = nonzers[relevant_idx]
        except:
            print(toks[i])
            print(nonzers)
            print(relevant_idx)
            print(i)
            raise ValueError("Something went wrong")
        nonzers = nonzers[0]
        nonzers = nonzers.item()
        end_idxs_raw.append(nonzers)
    end_idxs = torch.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALL_SEM = [
    "N",
    "R",
    "C",
    "end",
]


def get_idx_dict(prompts, tokenizer, prepend_bos=False, toks=None):
    [N1_idxs, ICL1L_idxs, ICL2L_idxs, R_idxs] = get_name_idxs(
        prompts,
        tokenizer,
        idx_types=["N", 'ICL1L', "ICL2L", 'R'], #["N", "R"],
        prepend_bos=prepend_bos,
    )
    
    end_idxs = get_end_idxs(
        prompts,
        tokenizer,
        name_tok_len=1,
        prepend_bos=prepend_bos,
        toks=toks,
    )

    # punct_idxs = get_word_idxs(prompts, [",", "."], tokenizer)

    return {
        # "R-1": R_idxs-1, # the 
        "R": R_idxs, # capital
        "N-1": N1_idxs - 1, # of
        "N": N1_idxs, # nation 
        "ICL1L": ICL1L_idxs,
        "ICL2L": ICL2L_idxs,
         #"N+1": N1_idxs + 1, # is
        "end": end_idxs,
        "starts": torch.zeros_like(end_idxs),
        # "punct": punct_idxs,
    }

class FactDataset:
    def __init__(
        self,
        prompt_type: Union[
            str, List[str]
        ],  # if list, then it will be a list of templates
        N=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        ioi_prompts_for_word_idxs=None,
        prepend_bos=False,
        manual_word_idx=None,
        counterfact=False,
        nation=None,
        add_prefix=0,
    ):
        """
        ioi_prompts_for_word_idxs:
            if you want to use a different set of prompts to get the word indices, you can pass it here
            (example use case: making a ABCA dataset)
        """

        if not (
            N == 1
            or prepend_bos == False
            or tokenizer.bos_token_id == tokenizer.eos_token_id
        ):
            warnings.warn(
                "Probably word_idx will be calculated incorrectly due to this formatting"
            )
        assert not (symmetric and prompt_type == "ABC")
        assert (
            (prompts is not None) or (not symmetric) or (N % 2 == 0)
        ), f"{symmetric} {N}"
        self.prompt_type = prompt_type


        if isinstance(prompt_type, list):
            self.templates = prompt_type
        elif prompt_type == 'NCNC':
            self.templates = NC_TEMPLATES.copy()
        else:
            raise ValueError(prompt_type)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        self.prompt_type = prompt_type

        if prompt_type == 'induction':
            self.ioi_prompts = []
            for temp_id in range(N):
                ioi_prompt = {}
                ioi_prompt["text"] = self.tokenizer.batch_decode(torch.randint(50000, (N, 200)))
                ioi_prompt["TEMPLATE_IDX"] = temp_id
                self.ioi_prompts.append(ioi_prompt)
        elif prompts is None:
            self.ioi_prompts, N = gen_prompt_uniform(  # a list of dict of the form {"text": "Alice and Bob bla bla. Bob gave bla to Alice", "IO": "Alice", "S": "Bob"}
                self.templates,
                NATIONS,
                CITIES,
                nc_dict=PAIRED_NC,
                N=N,
                symmetric=symmetric,
                prefixes=self.prefixes,
                counterfact=counterfact,
                all_same=(prompt_type == "all_same"),
                passed_nation=nation,
                tokenizer=self.tokenizer,
            )
        else:
            # assert N == len(prompts), f"{N} and {len(prompts)}"
            self.ioi_prompts = prompts
            N = len(prompts)

        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        small_groups = []
        for group in self.groups:
            if len(group) < 5:
                small_groups.append(len(group))
        if len(small_groups) > 0:
            warnings.warn(
                f"Some groups have less than 5 prompts, they have lengths {small_groups}"
            )

        self.sentences = [
            prompt["text"] for prompt in self.ioi_prompts
        ]  # a list of strings. Renamed as this should NOT be forward passed
        
        # self.templates_by_prompt = []  # for each prompt if it's ABBA or BABA
        # for i in range(N):
        #     if self.sentences[i].index(self.ioi_prompts[i]["N"]) < self.sentences[
        #         i
        #     ].index(self.ioi_prompts[i]["C"]):
        #         self.templates_by_prompt.append("NC")
        #     else:
        #         self.templates_by_prompt.append("CN")

        # print(self.ioi_prompts, "that's that")
        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.ioi_prompts
        ]

        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
            torch.int
        )

        # print(self.tokenizer(texts, padding=True))
        # input(self.tokenizer.decode([15597, 287, 2000,]))
        # tokenizer没加bos也没加eos

        if ioi_prompts_for_word_idxs is None:
            ioi_prompts_for_word_idxs = self.ioi_prompts
        
        self.word_idx = get_idx_dict(
            ioi_prompts_for_word_idxs,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )

        self.prepend_bos = prepend_bos
        if manual_word_idx is not None:
            self.word_idx = manual_word_idx

        self.sem_tok_idx = {
            k: v for k, v in self.word_idx.items() if k in ALL_SEM
        }  # the semantic indices that kevin uses
      
        self.N = N
        
        self.max_len = max(
            [
                len(self.tokenizer(prompt["text"]).input_ids)
                for prompt in self.ioi_prompts
            ]
        )
        
        # in-weight answer
        tokenizer_offset = 0
        if 'opt' in self.tokenizer.name_or_path:
            tokenizer_offset = 1
             
        self.N_tokenIDs = [
            self.tokenizer.encode(" " + prompt["N"])[0 + tokenizer_offset] for prompt in self.ioi_prompts
        ]

        self.IW_tokenIDs = [
            self.tokenizer.encode(" " + prompt["IW answer"])[0 + tokenizer_offset] for prompt in self.ioi_prompts
        ]

        self.R_tokenIDs = [
            self.tokenizer.encode(" " + prompt["R"])[0 + tokenizer_offset] for prompt in self.ioi_prompts
        ]
        
        # in-context answer
        # self.IC_tokenIDs = [
        #     self.tokenizer.encode(" " + prompt["C"])[0] for prompt in self.ioi_prompts
        # ]

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

    def add_a_shot(self):
        from copy import deepcopy
        prompts = []
        for i in range(self.N):
            prompt = deepcopy(self.ioi_prompts[i])
            
            while True:
                f1 = rd.choice(NATIONS)
                if not (f1 in prompt['text']):
                    break

            # prompt['text'] = f"The capital of {f1} is {PAIRED_NC[f1]}. " + prompt['text']
            prompt['text'] = f"{f1}'s capital is {PAIRED_NC[f1]}. " + prompt['text']
            prompts.append(prompt) 

        few_shot_dataset = FactDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=prompts,
            prefixes=self.prefixes,
            ioi_prompts_for_word_idxs=None,
            prepend_bos=self.prepend_bos,
        )
        return few_shot_dataset

    def gen_flipped_prompts(self, flip):
       
        assert isinstance(flip, tuple) or flip in [
            "prefix",
        ], f"{flip} is not a tuple. Probably change to ('IO', 'RAND') or equivalent?"

        if flip == "prefix":
            flipped_prompts = flip_prefixes(self.ioi_prompts)
        else:
            # TODO
            word_idx = self.word_idx
            if flip[1] == "RAND" and flip[0] in [
                    "N",
                    "C",
                ] + NATIONS + CITIES:
                flipped_prompts = gen_flipped_prompts(self.ioi_prompts, NATIONS, CITIES, flip)
            elif flip[0] in [
                    "N",
                    "C",
                ]:
                from copy import deepcopy
                flipped_prompts = deepcopy(self.ioi_prompts)
                for prompt in flipped_prompts:
                    prompt['text'] = prompt['text'].replace(prompt[flip[0]], flip[1])
                    prompt[flip[0]] = flip[1]
            elif flip[0] == "R":
                from copy import deepcopy
                flipped_prompts = deepcopy(self.ioi_prompts)
                for prompt in flipped_prompts:
                    prompt['text'] = prompt['text'].replace(prompt['R'], flip[1])
                    prompt['R'] = flip[1]
            elif (flip[0] == 'prefix' and flip[1] == '') or (flip[0] == 'all_same'):
                flipped_prompts = gen_flipped_prompts(self.ioi_prompts, NATIONS, CITIES, flip)
                texts = [
                    (self.tokenizer.bos_token if self.prepend_bos else "") + prompt["text"]
                    for prompt in flipped_prompts
                ]
                new_toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(
                    torch.int
                )
                word_idx = get_idx_dict(
                        flipped_prompts,
                        self.tokenizer,
                        prepend_bos=self.prepend_bos,
                        toks=new_toks,
                    )
                # input(word_idx)

        flipped_ioi_dataset = FactDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,
            prefixes=self.prefixes,
            ioi_prompts_for_word_idxs=flipped_prompts if flip[0] == "RAND" else None,
            prepend_bos=self.prepend_bos,
            manual_word_idx=word_idx,
        )
        return flipped_ioi_dataset

    def copy(self):
        copy_ioi_dataset = FactDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.ioi_prompts.copy(),
            prefixes=self.prefixes.copy()
            if self.prefixes is not None
            else self.prefixes,
            ioi_prompts_for_word_idxs=self.ioi_prompts.copy(),
        )
        return copy_ioi_dataset

    def __getitem__(self, key):
        sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = FactDataset(
            prompt_type=self.prompt_type,
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks


# tests that the templates work as intended
if __name__ == "__main__":
    f_dataset = FactDataset(
        prompt_type="all_same",
        N=100,
        prepend_bos=False,
        counterfact=False,
        nation='China',
    )
    input(f_dataset.tokenized_prompts)

    d1 = FactDataset(N=5, prompt_type='NC_mixed', counterfact=False)
    input(d1.tokenized_prompts)

    # d2 = FactDataset(N=5, prompt_type='NC_mixed', counterfact=False)
    # input(d2.tokenized_prompts)

    # d3 = FactDataset(N=5, prompt_type='CNNC')
    # input(d3.tokenized_prompts)

    # d4 = FactDataset(N=100, prompt_type='NCNC')
    # input(d4.tokenized_prompts)
    
    # abc_dataset_all = (
    #     d1.gen_flipped_prompts(("C", "RAND"))
    #     .gen_flipped_prompts(("N", "RAND"))
    #     .gen_flipped_prompts(("N1", "RAND"))
    # )
    # input(abc_dataset_all.tokenized_prompts)
    # abc_dataset_fact = FactDataset(N=100, prompt_type='NC_mixed', counterfact=False)