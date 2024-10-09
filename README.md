Latest update last.

16.9.2024:

Generative AI, such as ChatGPT, is not very good at producing palindromes, even though the models understand the
probability related to the proximity of different words. To learn something new, I wanted to test whether a machine
learning (ML) model could be developed to automatically generate palindromes.

From a human perspective, creating a palindrome feels natural. We often start with a word, for example ‘ISO,’ and
think about how letters could be added to it so that, when read backward, it would form a meaningful word.
At this point, we can add the letter T and get the word ‘ISOT,’ which backward is ‘TOSI.’ However, this is not
yet a palindrome because the words are not completely identical. A human would look for a word starting
with the letter T and might come up with the word ‘TILA.’ This would give us ‘ISO TILA,’ which backward is
‘ALI TOSI,’ thus forming a palindrome. The process requires continuing and modifying the word until we find a
 symmetrical whole. This human process is intuitive and dynamic, but challenging for a machine.


The Limitations of Machine Learning in Creating Palindromes

Machine learning, such as LSTM models (long short-term memory), learns to recognize probabilities and dependencies in
sequences. However, in the case of palindromes, it’s about strict symmetry — each letter or word mirrors itself in
reverse order. LSTM does not naturally handle such a structural requirement. While machine learning can help find
probable word combinations, it does not inherently recognize the strict structure required for symmetry.

First Attempts: Combining Random Words

I approached the problem with a simple programming test where I tried randomly combining words (
adjectives, verbs, nouns) to see if they would form palindromes. However, this proved inefficient,
as finding a palindrome through random sampling is rare. During practical tests, I got results consisting
of random words:

```
Sentence 'laiskansitkeä kevytsora heilahdella' is not palindrome.
Sentence 'eklektinen sinologia tuurata' is not palindrome.
Sentence 'urheilullinen panna murahtaa' is not palindrome.
```

This approach was not effective, as random combinations do not lead to a symmetrical structure.

A Learning Model for Recognizing Palindromes

I also began by building a learning model, where I fed the neural network words and sentences that were either
 palindromes or non-palindromes. The goal of the model was to learn to distinguish palindromic sentences from
 non-palindromic ones. This involved cleaning the words, removing punctuation, and finally analyzing the sequences.

```
def is_palindrome(text):
    return text == text[::-1]
```

 However, based on these experiments, it became clear that machine learning struggles to recognize symmetry.
 The task of the machine is to learn probabilities, but the symmetry required for palindromes is more complex
 and requires traditional logical verification algorithms.

Hybrid Model: Combining Machine Learning and Logic

An interesting solution could be a hybrid model that uses both machine learning and logical verification.
For example, the machine could suggest word combinations based on word frequency and context, and then an
algorithm would verify if the resulting sentence is symmetrical. This approach could leverage machine learning
to speed up suggestions, while the final verification would be rule-based.

Experiments and Observations:

Limitations of Machine Learning: Testing the LSTM model for generating palindromes did not yield the desired
results because the model is not sensitive to reverse structures. The model could identify partial palindromes
or give false positives.

Word-based Approach: Experimenting with randomly selected words was not an effective way to create palindromes.

Generating a palindrome requires a specific structure that cannot be achieved through random combinations.

Hybrid Model Could Be Promising: By combining machine learning and logical verification, we can develop a
method where the machine suggests probable word combinations, and then a logical algorithm checks if the
sentence is a palindrome.

Through these tests, it is clear that while AI and machine learning offer many possibilities,
generating palindromes still requires more specialized methods. AI can suggest word combinations,
but the unique nature of palindromes requires rule-based structural validation. Machine learning can
help accelerate the process, but creating a palindrome based solely on probabilities is challenging.

17.9.2024:

Step 1: Starting with a Word

The process begins with a word, such as saippua. The goal is to turn this word into a palindrome.
A palindrome, of course, is a word or phrase that reads the same backward as forward.

Step 2: Creating a Mirror Image

Next, I generate a mirror image of the word, which is technically an anagram of the word’s reverse.
However, this mirrored version is not yet a palindrome. At this stage, we have something like saippuaauppias,
which isn’t symmetrical.

Step 3: Testing Symmetry with Letter Insertion

This is where things get interesting. I take the mirrored word and try inserting letters into the middle,
terating through the Finnish alphabet (a-ö). Every single insertion produces a symmetrical word, but most of
these words will not be meaningful.

Step 4: Testing Against Word Lists

Now, the challenge is to ensure that the symmetrical word forms valid, meaningful words. To do this, the
program tests the generated symmetrical words against predefined word lists. For example, when inserting the
letter k in the middle of saippuaauppias, the resulting word is saippuakauppias. A quick lookup against my word
lists verifies that both saippua and kauppias are valid words, making saippuakauppias a palindrome!

Step 5: Expanding the Palindrome Knowledge

Currently, my program’s class, FEEDER, identifies existing palindromes like ämmä and isi in Finnish words.
These recognized palindrome words can be inserted in the middle of generated words to form even longer palindromes.
This gives the generator a solid base of symmetrical words to work with. However, I thought that if I add to the middle
words beginning with same letter which fitted to the middle, perhaps we find more palindromes. And we really did!

```
Palindrome-checker
1. Input word and check
2. End
Choose (1 or 2): 1
Input word: saippua
Found first: saippuakauppias
... checking if we can extend ...
Found! : saippua kaataa kauppias
Found! : saippua koko kauppias
Found! : saippua kakka kauppias
Found! : saippua kala kauppias
Found! : saippuakala kapakala kauppias
Found! : saippuakala kutukala kauppias
Found! : saippua kalla kauppias
Found! : saippua kama kauppias
Found! : saippua kana kauppias
Found! : saippua kanna kauppias
Found! : saippua kanuuna kauppias
Found! : saippua kappa kauppias
Found! : saippua kara kauppias
Found! : saippuakara kakara kauppias
Found! : saippua kasa kauppias
Found! : saippua kassa kauppias
Found! : saippua kili kauppias
Found! : saippua kiniini kauppias
Found! : saippua kippi kauppias
Found! : saippua kiri kauppias
Found! : saippua kitti kauppias
Found! : saippua kivi kauppias
Found! : saippuakivi kuukivi kauppias
Found! : saippua kiwi kauppias
Found! : saippua koho kauppias
Found! : saippua kokko kauppias
Found! : saippua koko kauppias
Found! : saippua kolo kauppias
Found! : saippua korohoro kauppias
Found! : saippua koto kauppias
Found! : saippua kukku kauppias
Found! : saippua kulu kauppias
Found! : saippua kumu kauppias
Found! : saippua kupu kauppias
Found! : saippua kuru kauppias
Found! : saippua kuttu kauppias
Found! : saippua kutu kauppias
Found! : saippua kuu kauppias
Found! : saippua kuuloluu kauppias
Found! : saippua kyky kauppias
Found! : saippuakyky kirikyky kauppias
Found! : saippua kyly kauppias
Found! : saippua kyty kauppias
Found! : saippua kyy kauppias
Found! : saippua käninä kauppias
Found! : saippua känä kauppias
Found! : saippua kässä kauppias
Found! : saippua köö kauppias
Found! : saippua kuuluu kauppias
Found! : saippua käytyä kauppias
Found! : saippua kaikkia kauppias
Found! : saippua kinni kauppias
Found! : saippua kysy kauppias
Found! : saippua ks kauppias

Palindrome-checker
1. Input word and check
2. End
Choose (1 or 2):
```

20.9.2024:

Created a PalindromeFeeder.py which generates palindromes from adjectives, substantives, verbs and long text from
book etc. sources. This do not take into account Finnish language complexities. Words are in basic form. Some of
complex words comes from the book sources.

Iteration of substantives took 18+ hours and found 14 368 palindromes restricted to less than 5 words
(file subs_palindromes.csv), adj_palindromes.csv contains 1177 palindomes, verb_palindromes.csv 1573 palindromes,
ext_palindromes.csv 4615 palindromes.

Palindromes are intended for machine learning purposes. With several tries I ended up using nltk-library.


27.9.2024:

With ntlk- library I was able to predict which word might fit to palindrome, but due to symmetry problem,
I did not develop this further. Perhaps punishment (-1) and appraisal (+1) approach with tensorflow + keras
might work, but I did not get that working so well with my knowledge.

I created FormWay.py, which is quite OK result for kind of game like thing. It could be developed further with
point system etc. Currently logic is simple: three input fields, check if anagram, if anagram, check towards
vocabulary does anagram make sense and if does, it is a palindrome! Prediction mechanism needs adjustment and
at least it is now simplier to test with different ML-models for predictions.

7.10.2024:

Added QT6 visualization and combined classes together.

Git do not allow long files to be uploaded. So, delete word2vec model files (both) and let script make them again.
File size of vectors_ngrams.npy is about 800 megabytes.
model file is about 1.5 Mb

See [Medium](https://medium.com/@jari.p.hiltunen/enhanced-palindrome-game-with-qt6-visualization-version-0-32-46d3930a9ccf) 

8.10.2024:

Created executable to build-directory with cx_Freeze. If you need to re-create executable, run 
python setup.py build at the root (where PalindromiPeli.py resides). However,
this did not work with other workstation (Linux) and I removed build and dist.

See [Medium](https://medium.com/@jari.p.hiltunen/distributing-your-python-application-a-guide-02e9405b774d)


9.10.2024:

With Windows nltk packages was downloaded to wrong location. Fixed path so that
now nltk tokenizers are downloaded under data-directory and model-files are kept
in data-directory. You can safely delete data/tokenizers and word2vec-model files,
because they will be recreated during startup.

With Windows I was able to compile Windows executable (available under windows-version).
For some unknown reason PyInstaller can not pack data-folder into distribution (see the spec file).
So, you need to copy the exe and data-directory to any location to your Windows PC and
first execution will download nltk library, extract tokenizers under data-directory and
then word2vec model files will be placed to data.
