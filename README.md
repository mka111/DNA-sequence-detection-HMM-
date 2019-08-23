# DNA-sequence-detection-HMM-

We will use the Viterbi algorithm on a hidden Markov model to detect C/G rich regions in a DNA sequence.

DNA sequences are made up of the four letters A, C, T and G (each letter corresponding to a particular nucleic acid). It turns out that, in the genomes of many organisms (including humans), the DNA letters are not distributed randomly, but rather have some regions with more A's and T's and some regions with more C's and G's. It turns out that C/G rich regions tend to have more genes, whereas A/T rich regions tend have more non-functional DNA. In this assignment we will analyze the genome of the bacterium Escherichia coli (E. coli), which is commonly found in the human gut.

We will use a hidden Markov model to detect C/G rich regions in a DNA sequence. We will use an HMM with two states, corresponding to low- and high-C/G regions respectively. The prior, transition and emission probabilities are given in the template file.

You need to implement two functions. HMM.logprob() calculates the probability of a particular sequence of states and characters. HMM.viterbi() uses the Viterbi algorithm to calculate the most likely sequence of states given a sequence of DNA characters. In other words, logprob() computes log(P(sequence, states)) and viterbi() computes argmax_states P(sequence, states).

The input file is "small.txt" which is a single sequence of letters A, C, T, G

Run the code using command:
```
python a3_template.py "small.txt" 
```

