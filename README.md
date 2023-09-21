# Experiments done during SERI MATS (Summer 2023)

## Relation to research write-ups

### `/refusal`

Activation steering with a "refusal vector" to cause llama-2-chat model to stop refusing to answer harmful questions.

- [Red-teaming language models via activation engineering](https://alignmentforum.org/posts/iHmsJdxgMEWmAfNne/red-teaming-language-models-via-activation-engineering)

### `/sycophancy`

Activation steering to modulate sycophancy in llama-2-chat and llama-2 base model.

- [Modulating sycophancy in an RLHF model via activation steering](https://alignmentforum.org/posts/raoeNarFYCxxyKAop/modulating-sycophancy-in-an-rlhf-model-via-activation)

- [Reducing sycophancy and improving honesty via activation steering](https://alignmentforum.org/posts/zt6hRsDE84HeBKh7E/reducing-sycophancy-and-improving-honesty-via-activation)

- [Understanding and visualizing sycophancy datasets](https://www.lesswrong.com/posts/ZX9rgMfvZaxBseoYi/understanding-and-visualizing-sycophancy-datasets)

### `/steering`

Activation addition experiments (pure act-adds from single forward passes)

- [Activation adding experiments with llama-7b](https://www.lesswrong.com/posts/w9yKQzyhsLJEZhvg9/activation-adding-experiments-with-llama-7b)

- [Activation adding experiments with FLAN-T5](https://www.lesswrong.com/posts/c38nAg23YTCzd7m8P/activation-adding-experiments-with-flan-t5)

### `/intermediate_decoding`

Logit-lens experiments (directly decoding intermediate activations by passing them through unembedding layer)

- [Decoding intermediate activations in llama-2-7b](https://www.lesswrong.com/posts/fJE6tscjGRPnK8C2C/decoding-intermediate-activations-in-llama-2-7b)

## Other directories

### `/data_generation`

- Code for generating LLM-generated datasets using gpt-4, 3.5 and Claude APIs

### `/probability_calibration`

- Early stage experiments to try and measure whether LLMs are aware of their internal uncertainty over a prediction

### `/unlearning`

- Early stage attempt at Google's [Machine Unlearning Challenge](https://blog.research.google/2023/06/announcing-first-machine-unlearning.html)

