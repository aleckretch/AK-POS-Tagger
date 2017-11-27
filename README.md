# AK-POS-Tagger
A Part-of-Speech (POS) tagger built using a visible Markov Model (MM) and a Maximum Entropy Markov Model (MEMM) in Python

# Usage

The MM considers the probability of known tokens to occur as any POS tag multiplied by the probability of any POS tag following the previously predicted POS tag. Unknown tokens consider the tag-to-tag probability multiplied by the unknown token entity probability to occur as any POS tag. Unknown tokens additionally consider the probability of tokens including certain lexical features to occur as any POS tag, including inclusion of a common suffix, hyphen, or first-character capital. The probability data is according to a pre-trained data file.

To use import the source file into your code: `import mm`

The MM takes in an array of tokens and returns an array of associated POS tags. Here is a sample use case:

```
m1 = MM.mm()
tokens = ["I", "enjoy", "eating", "sweet", "apples", "dipped", "in", "warm", "honey", "."]
tags = m1.get_pos_tags(tokens)
# tags: ["PRP", "VBP", "VBG", "JJ", "NNS", "VBD", "IN", "JJ", "NN", "."]
```

The default data file used is *memm-model.txt*, but an alternative data file can be selected by including the *model_path* parameter, like such:

```
m1 = MM.mm(model_path="other-memm-data-file.txt")
```

The MEMM considers tokens in a +/- 2 token window of a prospective token (including the prospective token). Each tag of each token is given a best-fitting vector based on the tokens around it. The best-fitting vectors are compared to the similarly-structured prospective token vector through a cosine similarity function to determine the most likely tag. The MEMM is trained via a multi-class perceptron. The MEMM runs much more slowly in comparison to the MM for large data sets and tends to perform less accurately. However, it may be useful for certain circumstances and is still available for testing.

Usage is identitcal for the MEMM as the MM.

# Training

The MM and MEMM can be trained, tuned, and tested against different data sets to attempt to produce better results for your needs. *dev.py* provides a command-line tool for these purposes. Here are some sample executions:

```
$ python3 dev.py -f dev.tagged -tu -mm -min 2
Overall correct: 113976 (0.945)
Overall incorrect: 6627 (0.055)
Unknown correct: 4389 (0.637)
Unknown incorrect: 2504 (0.363)
$ python3 dev.py -f train.tagged -tr -mm
Overall correct: 181552 (0.946)
Overall incorrect: 10417 (0.054)
Unknown correct: 6096 (0.659)
Unknown incorrect: 3161 (0.341)
$ python3 dev.py -f test.tagged -te -mm -m mm-model.txt
Overall correct: 52829 (0.933)
Overall incorrect: 3771 (0.067)
Unknown correct: 3805 (0.736)
Unknown incorrect: 1365 (0.264)
$ python3 dev.py -f dev.tagged -tu -memm -fe 30 -min 2 -low -e 10
Overall correct: 115464 (0.957)
Overall incorrect: 5139 (0.043)
Unknown correct: 5333 (0.539)
Unknown incorrect: 4560 (0.461)
$ python3 dev.py -f train.tagged -tr -memm -s
Overall correct: 186376 (0.971)
Overall incorrect: 5593 (0.029)
Unknown correct: 8518 (0.642)
Unknown incorrect: 4740 (0.358)
$ python3 dev.py -f test.tagged -te -memm -m memm-model.txt
Overall correct: 50836 (0.898)
Overall incorrect: 5764 (0.102)
Unknown correct: 2919 (0.504)
Unknown incorrect: 2868 (0.496)
```
