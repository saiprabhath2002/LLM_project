import torch
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import brown
from collections import Counter
import random
from googletrans import Translator
from transformers import AutoModelForCausalLM,AutoTokenizer, AutoModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available else 'cpu'


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
vocab = tokenizer.vocab
vocab_words = vocab.keys()
nltk.download('brown')
# Get word frequencies from the Brown corpus
words_in_brown = brown.words()
brown_set = set(words_in_brown)
word_counts = Counter(words_in_brown)
words_long = [word for word in brown_set if len(word)>=3 and word in vocab_words]

translator = Translator()

# import time

def translate_word(eng_word, lang_list):
    word_list = []
    for lang in lang_list:
#         print(lang)
        try:
            translated_word = translator.translate(eng_word, src="en", dest=lang)
    #             print(translated_word.text)
            word_list.append(translated_word.text)
            # time.sleep(1)
        except:
            continue
    
    if word_list is not None and len(lang_list) == len(word_list):
        return word_list
    return None

lang_list = ['fr', 'es', 'zh-CN', 'hi']





words_df = pd.DataFrame(columns=['en'] + lang_list)

# for i in range(50):
# for random_word in tqdm(words_long):
#     if not random_word.isdigit():
#         word_list = translate_word(random_word, lang_list)
#         if word_list is not None:
#             word_list = [random_word] + word_list
#             new_row_df = pd.DataFrame([word_list], columns=words_df.columns)
#             words_df = pd.concat([words_df, new_row_df], ignore_index=True)
            
#         if words_df.shape[0]%50>0 and words_df.shape[0]%50 == 0:
#             print(f'Words added: {words_df.shape[0]}/{5000}')
#     #     if words_df.shape[0] == 100:
#     #         break
# words_df.to_csv('words_1k.csv', index=False)

words_df = pd.read_csv('words_1k.csv')


words_df = words_df.loc[~words_df['en'].str.isdigit()]
words_df = words_df.reset_index(drop=True)
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7")



embedding_layer = model.transformer.word_embeddings.to(device)
# embedding_layer = model.transformer.to(device)

output_list = []

with torch.no_grad():
    for i in tqdm(range(len(words_df))):
        words = words_df.loc[i].tolist()
        input_list = [tokenizer(word, return_tensors='pt') for word in words]
        input_list = [d['input_ids'][0].to(device) for d in input_list]
        outputs = [torch.mean(embedding_layer(input.to(device)), dim=0) for input in input_list]
        output_list.append(torch.stack(outputs))
        
output_tensor = torch.stack(output_list)

lang_name = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'zh-CN': 'Chinese',
    'hi': 'Hindi',
    
}



complete_lang_list = ['en'] + lang_list
# cs_scores = np.zeros((5,5))

for i in range(5):
    for j in range(i+1, 5):
        tensor_a = output_tensor[:, i, :].squeeze(1)
        tensor_b = output_tensor[:, j, :].squeeze(1)
        cosine_similarity_matrix = torch.nn.functional.cosine_similarity(tensor_a.unsqueeze(1), tensor_b.unsqueeze(0), dim=2)
        cosine_similarity_matrix_np = cosine_similarity_matrix.detach().cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(cosine_similarity_matrix_np, annot=False, cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
        # plt.title("Cosine Similarity Matrix Heatmap")
        plt.xlabel(lang_name[complete_lang_list[i]])
        plt.ylabel(lang_name[complete_lang_list[j]])
        plt.savefig(complete_lang_list[i] + '_' + complete_lang_list[j] + '.png', bbox_inches='tight', pad_inches=0)
        # plt.show()










