# How Large Language Model works: 

## LLM are Autocomplete trained systems
So most of us are familiar with ChatGPT. It uses LLM behind the scene. We give a prompt and it responds back with an answer. 
When you give a prompt or instruction to LLM it generates text - one word at a time. It updates the response by adding that word and predicts for the next closest word. This process repeats until it finds a complete sentence.
It is just a `autocomplete system` trained to predict the next word in a sequence based on the words before it. You can find a similiar thing in your phone keypad which autocompletes your sentence. But in LLM's case it is backed by a huge dataset with trillions of words. LLMS are trained to predict the words from the dataset. 

## How does LLM's learn: 

You cannot teach a computer what a dog is. It can only understand numbers. So how does it interpret ?
So each word has unique number called token. For example, "cat" = 17, "dog" = 42, "banana" = 99. This process is called `tokenization` where each entity is called as `token` and the number representing it is called as `token-id`.

| Token   | Token ID |
|---------|----------|
| cat     | 17       |
| dog     | 42       |
| banana  | 99       |

LLMS does not strictly follow one word is equal to one token. Instead a single word might get split into multiple tokens. 
**NOTE**: For the sake of understanding we are using one word as one token. In reality it could be a full word dog, part of the word ( ing in running ), punctuation (,  or . ) or even a single character in some cases. 

The goal here is to make computer understand that:
- CAT and DOG are similiar (both are animals)
- KING and QUEEN are related ( both are royalty )
- APPLE and CAR are totally different

Now how to make the computer understand this ?
Word Embeddings

## Word Embeddings

To understand the tokens and the relationship between them, each token ID is mapped to an array of numbers called an `embedding vector`. These vectors capture the meaning and context of each token. 
For example: 

| Token   | Token ID | Embedding Vector     |
|---------|----------|----------------------|
| cat     | 17       | [0.8, 0.8]           |
| dog     | 42       | [0.7, 0.7]           |
| banana  | 99       | [-0.8, 0.7]          |

If we plot this in a 2D graph for our understanding, you'd see that cat and dog are close together and banana is far away.
<br><br><img src="images/word_embeddings.png" width="600"/><br>

The vectors for cat and dog are similar, so the model knows they are related ( both are animals )
The vector for banana is different, so the model knows it is not related to cat or dog ( its a fruit )

LLMs use thousand of dimensions like 3D, 4D or even 1000D map, to capture super detailed relationships.

| Token   | Token ID | Embedding Vector (Made up numbers for illustration)                         |
|---------|----------|-----------------------------------------------------------------------------|
| cat     | 17       | [0.8, 0.2, 0.1, 0.5, 0.9, 0.3, 0.4, 0.6, 0.2, 0.8, 0.1, 0.7, 0.4, 0.5, 0.3] |
| dog     | 42       | [0.7, 0.3, 0.2, 0.4, 0.8, 0.1, 0.6, 0.5, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7] |
| banana  | 99       | [0.1, 0.9, 0.5, 0.3, 0.2, 0.8, 0.4, 0.6, 0.7, 0.1, 0.5, 0.9, 0.3, 0.2, 0.4] |

## How embeddings are generated

LLM's uses a model called `pretraining`. Here's a simplified version of how it works:

- LLM scans multiple source of texts like wikipedia, social media or books and lists all the unique tokens. Lets say it finds 50000 tokens. 
- Each token has a random `embedding`, like [0.1, -0.4, 0.7] for “cat” .
- From the input text a sentence is picked, for example "The cat sat on a mat"
- The LLM is made to predict the missing word from the sentence "The cat sat on a __"
- LLM uses the current embedding of the words "The", "cat", "sat", "on", "the" to predict the next token. If the prediction goes wrong ( Lets say it predicts the word "dog" ), it tweaks the embedding number in such a way that it predicts the word "mat"
- This process repeats multiple times so that the LLM finds the missing token by adjusting the embeddings.
- This process also includes multiple sentences which helps LLMs to get embeddings of all the available tokens. 

   <img src="images/word2vec_animation.gif" width="500"/><br>

**NOTE:**

- Pretraining is expensive
- Data can be scrapped from web for pre-training but raw web data is noisy, contains duplicates, low quality texts, html tags and irrelevant information. It requires extensive filtering before using the data. Efficient way is to use curated datasets which are already cleaned and organized like [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

## How do we know when Pre-training is complete: 

Pre-training is unsupervised. No human intervention is required during the pre-training. 
So the actual model is hidden from the LLM and it is allowed to predict the next token. 
For Example: If the LLM is allowed to predict the next word in the sentence "The cat sat on the ___ " . Lets say the cat predicts "pond" , which is definetly wrong and it is a big error. The actual word is "mat". So this error is considered as high `loss` where loss is the metrics which is used to measure how far is the model's prediction from the next available token. When it predicts "mat" then the loss is zero. 
Lower loss means better predictions. Therefore, if the loss values are decreasing over a period of time then the model is learning. 

Loss formula:
```
Loss = -log(Predicted Probability of True Next Token)
```
This means: if the model gives 80% probability to the correct word, loss = -log(0.8) ≈ 0.22. If it gives 100% probability, loss = -log(1.0) = 0.

## The base model

After pre-training the model that is generated is known as base model, foundational model or pretrained model.
Pre-training is when LLM reads the entire internet. It learns: 
- The missing word in a sentence.
- How to form a sentence
- That "cat is an animal" is more likely than "cat is a dog"

## Fine-Tuning








