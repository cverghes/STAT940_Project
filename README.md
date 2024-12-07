Project 5: Symbolic Regression with Diffusion Models

Problem:
Symbolic regression aims to discover mathematical expressions that best fit a given dataset. Traditional techniques, such as genetic algorithms, are often computationally intensive due to the complexity of mathematical expressions. Transformer-based models, like SymbolicGPT, have introduced a promising alternative by generating formulas similarly to text sequences. However, diffusion models, which have been successfully used for image generation and more recently adapted for text generation, offer a new opportunity for symbolic regression.

Recent Developments:
Diffusion models, originally applied to generate high-quality images, are now showing potential in text generation tasks. Since symbolic regression leverages a language of mathematical formulas, a diffusion-based approach—if effective for natural language—could also be applied to mathematical expressions. Unlike autoregressive models that generate one token at a time, diffusion models consider the entire sequence holistically, which could potentially result in better performance and more stable outputs.

Idea:
This project will explore how diffusion models, which avoid sequential token dependencies, can be adapted to symbolic regression. The aim is to assess whether diffusion-based symbolic regression can outperform current approaches in terms of efficiency and accuracy by generating mathematical expressions as complete structures rather than token-by-token.

Objective:
The primary objective is to try this new approach and compare it with existing methods. Since diffusion models do not rely on autoregressive generation, they could potentially improve results by avoiding the limitations associated with sequential token generation. This project will benchmark the diffusion-based model against transformer-based approaches and analyze its performance on accuracy, efficiency, and data handling.

Key Papers:

SymbolicGPT: A Generative Transformer Model for Symbolic Regression
Diffusion-LM: Improving Controllable Text Generation
Empowering Diffusion Models on the Embedding Space for Text Generation
SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers
End-to-end symbolic regression with Transformers
