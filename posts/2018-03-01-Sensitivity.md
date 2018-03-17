---
title: Sensitivity Analysis
---

hm so if i'm not mistaken, CV perturbs by backpropagating the error to some target label wrt the input image.
and yeah this sounds fun, kind of seems like there's room for a "neural edit distance"
(which would maybe end up looking like some kind of semantic bleu?)
or some other kind of metric learning for the fluent change part,
although that setup kind of sounds like NLI so it might be worth looking at some approaches there
+ the hallucination of inputs through GD or some discrete search method that utilizes gradients wrt inputs,
+ let me read the the jia+liang paper more closely.
+ i think i'm more interested in methods that have complete access to models (gradients, predictions) than model-agnostic ones,
+ were you more excited about jia+liang approach than the CV GD-based approach?
+ i'll get back to you on formalizing, i think the sensitivity / credit assignment problem is pretty important from a debugging standpoint.
+ i guess the jia+liang approach could basically be seen as perturbing the input 
 
 
# Related Work
- Jia + Liang
- Influence functions?
- Vision / adversarial?
- Integrated gradients, etc
- 
 

- https://openreview.net/pdf?id=H1BLjgZCb
