import os
from collections import Counter
import matplotlib.pyplot as plt

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().split()

train_files_normal = ['train_N_I.txt', 'train_N_II.txt', 'train_N_III.txt']
train_files_spam = ['train_S_I.txt', 'train_S_II.txt', 'train_S_III.txt']

train_normal = []
for file in train_files_normal:
    train_normal.extend(read_file(file))

train_spam = []
for file in train_files_spam:
    train_spam.extend(read_file(file))

counts_normal = Counter(train_normal)
counts_spam = Counter(train_spam)

total_words_normal = sum(counts_normal.values())
total_words_spam = sum(counts_spam.values())

prob_words_normal = {word: (counts_normal[word] + 1) / (total_words_normal + len(counts_normal)) for word in counts_normal}
prob_words_spam = {word: (counts_spam[word] + 1) / (total_words_spam + len(counts_spam)) for word in counts_spam}

prior_prob_normal = 0.73
prior_prob_spam = 0.27

def classify_email(email_words):
    prob_normal = prior_prob_normal
    prob_spam = prior_prob_spam
    
    for word in email_words:
        if word in prob_words_normal:
            prob_normal *= prob_words_normal[word]
        else:
            prob_normal *= 1 / (total_words_normal + len(counts_normal))
        
        if word in prob_words_spam:
            prob_spam *= prob_words_spam[word]
        else:
            prob_spam *= 1 / (total_words_spam + len(counts_spam))
    
    if prob_normal > prob_spam:
        return 'Normal'
    else:
        return 'Spam'

test_files = ['testEmail_I.txt', 'testEmail_II.txt']

for test_file in test_files:
    test_email_words = read_file(test_file)
    classification = classify_email(test_email_words)
    print(f"{test_file} is classified as {classification}")

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.bar(list(counts_normal.keys()), list(counts_normal.values()))
plt.title('Word Frequencies in Normal Emails')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
plt.bar(list(counts_spam.keys()), list(counts_spam.values()))
plt.title('Word Frequencies in Spam Emails')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
