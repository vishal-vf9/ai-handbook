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

Key words: 

- **Weights**: So tokens attend to each other and Attention Weights decide which token matter more for a given context. These dynamics are not store in embeddings. They are calculated on the fly.  In “The cat chased the dog,” the model learns to give more weight to “chased” when shaping “cat”’s embedding, so it reflects the action.

- **Context Window**: 









