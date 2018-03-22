---
title: Meeting Notes on Phrase Translation
---

# Summary

## Dataset

We train and evaluate our model on the IWSLT '14 De-En dataset.

TODO: add summary statistics

## Models

Our baseline is similar to Seq2Seq w/ attention, as in Luong et al. 2015.
One difference, however, is that we use a bidirectional encoder.

## Phrase Construction

Phrases are collected from the training corpora by selecting ngrams, in our case of lengths 2 through 10,
that appear more than 10 times. 
We then essentially give them their own embeddings. This can be done by giving it an embedding
that is fully independent of its constituents, or by composing the embeddings of its constituent words.

## Performance

Currently we are yet to match baseline performance.

| Model                   | Train PPL | Valid PPL | Valid Acc | Test BLEU | $\Delta$ Baseline |
| :---------------------- | --------: | --------: | --------: | --------: | ----------------: |
| Baseline                |        NY |       7.7 |      62.8 |     30.52 |                -  |
| Phrase2Word             |        NY |       9.0 |      60.0 |     27.17 |             -3.35 |
| **C**Phrase2Word        |        NY |       8.1 |      61.8 |     28.57 |             -1.95 |
| Word2Phrase             |        NY |      47.7 |      42.0 |     24.43 |             -6.09 |
| Word2**C**Phrase        |        NY |      44.7 |      42.1 |     27.10 |             -3.42 |
| **C**Phrase2**C**Phrase |        NY |      50.0 |      40.8 |     25.80 |             -4.72 |

Table 1: The performance of each model on validation (conditional language modeling)
as well for regenerating the test corpus. The training perplexities are Not Yet filled in.
Speed comparisons forthcoming.

In the above table, the baseline model is a traditional word to word model.
All other models are variants of Word, Phrase, or CPhrase on the source or target side,
indicated by **Source**2**Target** respectively.
The phrase model, indicated by "Phrase," gives each individual phrase its own word embedding
independent of its constituents.
A prefix of "**C**" indicates that the model composed a phrase's constituent word embeddings
with a BLSTM to produce a single embedding for that phrase.

# Source Phrase to Target Word Analysis

## Performance discrepancy

We posit that the reduction in performance stems from the parameterization of the attention distribution.
Assuming that our baseline model is strong, one way to achieve the same performance
using a phrase model would be to have it remain "faithful" to the baseline's alignments.
We define a "faithful" phrase model to be one that assigns attention mass to phrases equal to the sum of
the scores of its constituents in the baseline model.

* Is high $D_{KL}$ correlated with high $\Delta_{PPL}$?

## Repeat phrase encodings (with N constituents) N times

| Model                 | Train PPL | Valid PPL | Valid Acc | Test BLEU | $\Delta$ Baseline | $\Delta$ no repeat |
| :-------------------- | --------: | --------: | --------: | --------: | ----------------: | -----------------: |
| Baseline              |       5.3 |       7.7 |      62.8 |     30.52 |                 - |                  - |
| **C**Phrase2Word      |       5.4 |       8.1 |      61.8 |     28.43 |             -2.09 |                  - |
| **C**Phrase**R**2Word |       5.2 |       8.0 |      61.9 |     29.36 |             -1.16 |              +0.93 |
<!---
| **RC**Phrase2Word |       7.9 |      62.2 |     29.88 |             -0.64 |              +1.31 |
-->
Table 2: Experiment results for repeating phrase encoder outputs N times.
For the two phrase experiments, I took the MEDIAN of three trials.

## Hypotheses and proposed experiments
1. The manual bias in the attention distribution helps learning initially.
    - Run on a larger dataset, if the compositional phrase model and compositional phrase repeat model
      achieve the same performance, then we know it was a small-data problem.
    - On an even smaller dataset, the difference should be even more pronounced.
      We can run on half of IWSLT14 and we would expect the performance of CP2W to degrade 
      faster than CPR2W.
    - We can try up-weighting the attention on non-phrase words.
      We would expect this to decrease performance if the hypothesis is correct.
      For the down-weighting and up-weighting of phrases, we can measure the difference
      between the sum of the attention on the constituents and the phrases themselves.
      For the baseline model, we can also try some random weights that we set at initialization
      and see if that affects training (sensitivity analysis for an attention prior).
    - We can gate the attention weights and initialize it to 1 and see if the model turns it off
      eventually. I'm not sure exactly what I would be expecting here, or whether any insights
      would be gained from this experiment.
2. The upweighting of phrases gives more gradient to the compositional model
    i.e. the compositional phrase rnn is gradient-impoverished in the CP2W model.
    - Try a per-parameter learning rate (i.e. give the compositional parameters
      double the learning rate).
    - Adaptive methods like Adam might be helpful.
    - I think again running on a larger dataset would be informative.
    - This is pretty heuristic, but the norm of the weights can be a good indicator of
      how many updates a parameter has received.
    - We can sum up the **total** attention received by phrases in the first epoch of learning.
      
<!---
Actually, this is identical to scaling the pre-softmax attention score by N. 
Currently working on this in order to preserve speed. 

### Why does repeating improve results?
We hypothesized that the attention function has trouble with decoding the encoder's counts since the
attention mechanism utilizes a dot product and the LSTM encoder does not encode count using magnitude.
<!---
We could test this by using sum-pooling and checking the attention statistics as below in Table 5.

Actually, it turns out this hypothesis was at most half correct. 
There seems to be correlation between the norm of the embedding and the length of the phrase.
However, we believe that the norm is also well correlated with the number of times a unigram 
appears in the corpus overall (since its weights are updated as many times as it appears in the corpus times the number of epochs),
and therefore the total number of occurrences is a confounder.

Maybe the norm of the phrase is actually just approximately the max norm of its constituents.
That would not be good...

|           | Max Norm | Mean Norm | Min Norm |
| :-------- | -------: | --------: | -------: |
| Unigrams  |     5.70 |      1.73 |     1.25 |
| Bigrams   |     5.77 |      4.06 |     2.67 |
| Trigrams  |     6.27 |      4.56 |     3.12 |
| Fourgrams |     6.49 |      4.95 |     3.81 |
Table 3: Norm statistics for embeddings in the **C**Phrase model.

|           | Max Norm | Mean Norm | Min Norm |
| :-------- | -------: | --------: | -------: |
| Unigrams  |     5.87 |      1.81 |     1.26 |
| Bigrams   |     5.28 |      3.88 |     2.64 |
| Trigrams  |     5.37 |      4.09 |     2.99 |
| Fourgrams |     5.96 |      4.24 |     3.22 |
Table 4: Norm statistics for embeddings in the **C**Phrase repeat model.

Maybe this is why ELMo is not so nice as an encoder.
We could test this by trying to extract both neighbouring words as well as their counts.

Alternatively, it could be a optimization/learning problem, and more attention allows faster learning.
Maybe comparing the grad norms of the non-scaled and scaled models could be elucidating,
but we wonder if that is a valid indicator. Rather than the attention score, we could scale the 
module specific learning rate.

### Well, why does it not help for Phrase2Word?

We need to think more about this.

## Paying attention to attention

We think the attention scores on constituents of a phrase must be summed together for this to be a fair comparison.

| Model             | Average H | Average $D_{KL}(Baseline // Model)$ |           |
| :---------------- | --------: | ----------------------------------: | --------: |
| Baseline          |           |                                     |           |
| Phrase2Word       |           |                                     |           |
| **R**Phrase2Word  |           |                                     |           |
| **C**Phrase2Word  |           |                                     |           |
| **RC**Phrase2Word |           |                                     |           |
Table 4: Attention statistics.

## Generation errors

* During generation, where were the worst errors?
  How do you quantify an error during generation?
-->
# Possibilities for next week
* Squeeze a little more perf out from source phrases
* Adding as composition function
* Try scaling the attention scores of phrases by N constituents
* Supervised attention
* Feed word vectors + phrase vectors
<!---
* Multiheaded attention
    * My reservation with this is that if it results in a performance increase,
      then it should be used in the baseline as well.
* Analysis on Phrase to Phrase
* Annotated heatmap in visdom
* Citations!
* Fusion in onmt and beam search
-->

# Todo 2/13
1. Upper bound: add embedding skip connections to attention in baseline brnn model
2. Lower bound: use position embeddings instead of word embeddings, something less powerful? to see if "overfitting"
3. Check dropout on phrase composition
4. Visualize random sample of attentions from valid
5. Target: check how many phrases could have occurred? Check how many phrases we get, and also whether any WORD sequences get phrases
6. Noncompositional output for target with compositional input
7. Length during beam search, need to use stack decoding
8. Multiheaded attention?
9. Why does adding embeddings result in worse performance?!

## Milestones

so i think based on the LSD paper, segmentations (either number or algorithm) can have a large effect on performance,
but any model changes I make would be orthogonal.
for next week I want to have a pipeline set up which makes it easy to run on different segmentations so
i can start experimenting with

1. LSD
2. sam's idea for multi-tasking the phrase encoder as BoW autoencoder
3. multi-headed attention.

for efficiency, I need to 
1. Distill to small model
2. Get the speedups with weight pruning? 
3. What are the speedups people use nowadays
