---
title: Compression and Phrases
---

# Segmentations
## BPE
The BPE algorithm is credited to [Philip Gage](http://dl.acm.org/citation.cfm?id=177910.177914)
and was popularized in NMT by [Sennrich et al.](https://arxiv.org/abs/1508.07909).

Define `fresh` to be a function that returns a fresh variable from a class
of strings that does not appear in the corpus.
For example, we could have `fresh` return elements in the language
$\{\#\sigma^+ \mid \sigma \in \{A,\ldots,Z\}\}$.

Byte pair encoding is pretty simple.
We merge the most frequent pair of symbols into a new symbol and apply
this recursively.
As for encoding a new corpus, we simply apply the merge operations in the order they were learned.

<pre id="learnbpe" style="display:none;">
\begin{algorithm}
\caption{The BPE algorithm}
\begin{algorithmic}
\procedure{LearnBpe}{numMerges, corpus}
\state merges $\gets$ []
\while{len(merges) $\leq$ numMerges}
    \state bigramCounts $\gets$ \call{CountBigrams}{corpus}
    \state merge $\gets$ \call{MostFrequent}{bigramCounts}
    \state \call{RemoveBigram}{corpus, merge, \call{Fresh}{}}
    \state append merge to merges
\endwhile
\return merges
\endprocedure
\procedure{ApplyBpe}{merges, corpus}
\for{merge \textbf{in} merges}
    \state corpus $\gets$ \call{ApplyMerge}{corpus, merge}
\endfor
\return corpus
\endprocedure
\end{algorithmic}
\end{algorithm}
</pre>

<script>
var el = document.getElementById("learnbpe")
var code = el.textContent;
var parentEl = el.parentElement;
var options = {
    lineNumber: true
};
pseudocode.render(code, parentEl, options);
//var htmlStr = pseudocode.renderToString(code, options);
//console.log(htmlStr);
</script>

## Google Word Pieces

The approach used in Google's
[word pieces](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
(and subsequently used in [translation](https://arxiv.org/abs/1609.08144) and [another speech system](https://arxiv.org/abs/1712.06855))
is similarly simple.
Rather than combining the most frequent bigram, they greedily combine bigrams that increase the
performance of an ngram language model.
This can be interpreted as a coordinate ascent technique.

# LSD

Following the spirit of the [Latent Sequence Decomposition](https://arxiv.org/abs/1610.03035) paper,
we will define the problem more formally.
The LSD paper is concerned with the following: given an input $\mathbf{x}\in\bX$ and
a target $\mathbf{y}\in\bY$. The paper postulates that a probabilistic decomposition
of the target sequence conditioned on both the input sequence and output sequence is superior
to a deterministic decomposition that is only a function of the output sequence, as in most cases.
Note that this is not exactly true for BPE with a joint vocabulary over source and target.

Rather than defining a deterministic $f:\bY\to\bZ$, since knowing the optimal segmentation of
$\mathbf{y}$ is a nontrivial task, they propose to model $\mathbf{y}$ by marginalizing over all 
segmentations $\mathbf{z}$.
$$
\begin{aligned}
\log p_\theta(\by\mid\bx) &= \log \sum_{\bz\in\bZ}p_\theta(\by,\bz\mid\bx)\\
&= \log\sum_{\bz\in\bZ}p(\by\mid\bz,\bx)p_\theta(\bz\mid\bx)\\
&= \log\sum_{\bz\in\bZ}p(\by\mid\bz)p_\theta(\bz\mid\bx).
\end{aligned}
$$
Since $p(\by\mid\bz)$ is deterministic, as one can simply concatenate or split segmentations,
it does not need parameters to model the distribution of $\by\mid\bz$.
They then perform approximate inference via beam search.
As for the gradient of $p(\by\mid\bx)$, they use the log derivative trick twice.
Recall the log derivative trick is $\nabla\log f(\bx) = \frac{\nabla f(\bx)}{f(\bx)}$,
which implies $f(\bx)\nabla \log f(\bx) = \nabla f(\bx)$.
The goal here is to express the derivative as the expectation of some function of $\bx$
wrt the distribution over $\bz\mid\by,\bx$.
The gradient is given by
$$
\begin{aligned}
\nabla_\theta\log p_\theta(\by\mid\bx)
&= \frac{1}{p_\theta(\by\mid\bx)}\nabla_\theta p_\theta(\by\mid\bx)\\
&= \frac{1}{p_\theta(\by\mid\bx)} \sum_{\bz}\nabla_\theta p_\theta(\by,\bz\mid\bx)\\
&= \frac{1}{p_\theta(\by\mid\bx)}\sum_{\bz} p(\by\mid\bz,\bx) \nabla_\theta p_\theta(\bz\mid\bx)\\
&= \frac{1}{p_\theta(\by\mid\bx)}\sum_{\bz} p(\by\mid\bz,\bx)p_\theta(\bz\mid\bx) \nabla_\theta \log p_\theta(\bz\mid\bx)\\
&= \frac{1}{p_\theta(\by\mid\bx)}\sum_{\bz} p_\theta(\by,\bz\mid\bx) \nabla_\theta \log p_\theta(\bz\mid\bx)\\
&= \sum_{\bz} \frac{p_\theta(\by,\bz\mid\bx)}{p_\theta(\by\mid\bx)} \nabla_\theta \log p_\theta(\bz\mid\bx)\\
&= \sum_{\bz} p_\theta(\bz\mid\by,\bx) \nabla_\theta \log p_\theta(\bz\mid\bx)\\
&= \mathbf{E}_{\bz\thicksim p_\theta(\bz\mid\by,\bx)}[ \nabla_\theta \log p_\theta(\bz\mid\bx) ].
\end{aligned}
$$
When sampling from $z_t\mid\bz_{<t},\by,\bx$ they assign probability 0 to invalid extentions.
This implies that they do not sample from $\bz\mid\by,\bx$.
As detailed here [in the author's thesis](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1762&context=dissertations):
$$
\begin{aligned}
\nabla_\theta\log p_\theta(\by\mid\bx)
&= \frac{1}{p_\theta(\by\mid\bx)}\sum_{\bz} \bm{1}(\by=\bz)p_\theta(\bz\mid\bx) \nabla_\theta\log p_\theta(\bz\mid\bx)\\
&= \sum_{\bz} \frac{p_\theta(\by,\bz\mid\bx)}{\sum_{\bz'}p_\theta(\by,\bz'\mid\bx)} \nabla_\theta\log p_\theta(\bz\mid\bx)\\
&= \sum_{\bz} \frac{\bm{1}(\bz = \by)p_\theta(\bz\mid\bx)}{\sum_{\bz'}\bm{1}(\bz' = \by)p_\theta(\bz'\mid\bx)}
    \nabla_\theta\log p_\theta(\bz\mid\bx)
\end{aligned}
$$
They compare all possible hypotheses together, including those that span multiple time steps.
They say this leads to bias a bias towards longer sequences in sampling,
since they compare "c", "ca", and "cat" all at the same time.
However, they found that despite their hypothesis that the sampling is biased towards longer
segmentations the model learns, without regularization, to only output single characters.
They then use an entropy regularization term (they also refer to it as an $\epsilon$-greedy
strategy), or a mixture with a uniform prior over outputs (they anneal the mixing coefficient over time).

At test time, they use beam search to arrive at a segmentation.
Since there is no reference $\by$ available, all $\bz$ hypotheses are considered.
No marginalization is actually performed, since it seems like only a single Monte Carlo sample is taken,
but I could be mistaken.
The marginal probability of $\by\mid\bx$ could probably be computed pretty easily, though.

In their experiments they take ngrams from $n\in\{2,3,4,5\}$ and use the $\{256,512,1024\}$ most
frequent based on occurrences in the training set.
A quick description of their baseline segmentation method: they use the same ngrams but instead
of having a distribution over segmentations, they simply take the largest valid segmentation at
each time step greedily from left to right.
Interestingly, they find that this "MaxExt" (max extension) decomposition hurts their performance
versus the character baseline.
They do not compare against BPE or wordpieces, which people have found success with.
In the previously mentioned [paper](https://arxiv.org/abs/1712.01769), they saw a slight increase,
or at least no drop in performance (more likely no drop) with wordpieces.
How could the segmentation scheme have such a large effect on performance?

## Comparison of my segmentation procedure to MaxExt
The ngrams are gathered in a similar way, where they take the K most frequent 2 through N grams.
Where MaxExt greedily chooses the longest valid extension, I instead iterate from N through 2
and substitute constituents with phrases using the segmentation that maximize the number of replaced constituents.

# Multiscale Sequence Dictionary Learning
The [paper](https://arxiv.org/abs/1707.00762) by Merrienboer et. al. uses 2,048 crossword pieces
on PTB and 16,384 crossword pieces on text8 and sees a bit of improvement over both.
The interesting part of this work is that they actually marginalize over all possible
decompositions $\bz$ of a sequence $\by$, whereas LSD uses a (or possibly multiple)
Monte Carlo samples. Less relevant (but still interesting) is this work's incorporation of marginalization 
into the actually RNN computation.

## Marginal likelihood calculations

<pre id="ml" style="display:none;">
\begin{algorithm}
\caption{Marginal Likelihood and Gradient (in more detail)}
\begin{algorithmic}
\procedure{MarginalLikelihood}{$\mathbf{y}, \mathcal{Z}, \mathbf{x}, p_\theta(\cdot)$}
\state $p(y_0\mid\mathbf{x}) = p_\theta(y_i\mid\mathbf{x})$
\for{$t$ \textbf{in} $1,\ldots,|\mathbf{y}|$}
    \state $p(y_0, \ldots, y_t) = 1$
\endfor
\return lol
\endprocedure
\procedure{gML}{$\mathbf{y}, \mathcal{Z}, \mathbf{x}$}
\for{merge \textbf{in} merges}
    \state corpus $\gets$ \call{ApplyMerge}{corpus, merge}
\endfor
\return corpus
\endprocedure
\end{algorithmic}
\end{algorithm}
</pre>

<script>
var el = document.getElementById("ml")
var code = el.textContent;
var parentEl = el.parentElement;
var options = {
    lineNumber: true
};
pseudocode.render(code, parentEl, options);
//var htmlStr = pseudocode.renderToString(code, options);
//console.log(htmlStr);
</script>

# Other compression schemes
Do they make sense for phrases?

# Analysis of BPE vs ngram

| Segmentation    | Train Length | Avg Sen Len | Avg unit len | Phrase Count | Atom Count |
| :-------------- | -----------: | ----------: | -----------: | -----------: | ---------: |
| Word            |      3273295 |        20.4 |            4 |            0 |    3273295 |
| Ngrams 42k      |      2036480 |        12.7 |            7 |       769282 |    1267204 |
| BPE (Joint 40k) |      4434910 |        27.7 |          3.5 |            0 |    4434910 |
| Xword BPE 32k   |      2801504 |        17.5 |          5.9 |       729233 |    2072277 |
| Xword BPE 64k   |      2583680 |        16.1 |          6.4 |       790425 |    1793261 |
| Xword 32k unk   |      2784290 |        17.4 |          5.9 |       729118 |    2055178 |
| Xword 64k unk   |      2574701 |        16.1 |          6.3 |       789141 |    1785566 |

Since the phrase counts are similar, we posit that both techniques (XBPE and greedy) result in somewhat similar ngrams,
which we checked by eye.

Suppose that BPE mainly merges subword units before getting to xword.
If we reorder the merge agenda so that all subword merges happen before phrases, we should get an upper bound on the number of subword units.
An approximate lower bound on the length of this pure subword corpus is given by the length of the BPE corpus.
We then note that the compression ratio from word to ngram is 1.6x, and from word to XBPE 32k is 1.2x.
However, when working with the subword upper bound, we get a compression ratio from BPE to XBPE 32k of 1.6x.

Rare words do not seem to be a large factor in the length expansion, as the corpus statistics are relatively unchanged after 
unk-ing words that occur fewer than 3 times and an inspection by eye of the vocabulary leads us to believe the two are 
quite similar.

Oddity in all BPE subword models: certain obvious constructions like "& apos ;" do not get merged, which is pretty surprising. 
Is this a bug?

The objective is minimize total emission length.
Given a super-type inventory $\Sigma$, the objective decomposes into
$$\mathcal{J}(\Sigma) = \sum_{x\in\Sigma} \textrm{len}(x) * \textrm{freq}(x).$$
We can define $\Sigma$ 

Is ngram the maximizer of objective $\mathcal{J}$?
