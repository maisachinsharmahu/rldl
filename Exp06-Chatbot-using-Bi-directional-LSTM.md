# Experiment 6: Chatbot using Bi-directional LSTM

> **File:** `exp6.py`  |  **Tools:** Python 3, NumPy, Matplotlib, TensorFlow / Keras

## 🎯 Aim

To build a chatbot using bi-directional LSTMs.

## 📘 Theory

A retrieval-based chatbot works in two steps. First a model finds the intent of the user's message, then the answer written for that intent is returned. A bi-directional LSTM suits the first step: one LSTM reads the sentence forwards and another backwards, so the representation of every word depends on the words before and after it. Max pooling over all positions then keeps the strongest signal, for example the word 'library', wherever it appears.

The chatbot of this experiment answers questions about college life in 8 topics (library, hostel, exam, placement, fees, greeting, thanks and an 'other' class for off-topic questions). A small data set has little variety, so it is enlarged by a simple augmentation: five copies of every training sentence are made, each with one random word removed. A fifth of the original questions is kept aside to measure how the bot does on questions it has never seen. The reply shows the intent, its probability and the second best intent.

- **Intent**: The goal behind a user's message, such as asking about fees.
- **Data augmentation**: Creating new training examples by slightly changing existing ones.
- **Held-out set**: Examples kept out of training to test generalisation.
- **Global max pooling**: Takes the maximum of every feature over the whole sentence.

### 📐 Formulas

$$ {h}_{t} = \left[{h\to }_{t} ; {h\leftarrow }_{t}\right] $$

$$ p = \operatorname{softmax}\left(W {h}_{T} + b\right) $$

## 🪜 Algorithm / Steps

1. Write the example questions of the 8 topics and the answer of each topic.
2. Keep 20 % of the questions aside and create 5 copies with a random word removed for each training question.
3. Build Embedding → Bidirectional(LSTM, return sequences) → GlobalMaxPooling → Dropout → Dense(softmax) and train it for 40 epochs.
4. Measure the accuracy on the held-out questions, retrain on all questions, and chat with the bot.

## 💻 Code

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random

import keras
import numpy as np
from keras import layers

keras.utils.set_random_seed(21)
random.seed(21)

faq = {
    "library": ["can i issue books for my friend", "what are the library rules", "library fine for late return", "is there wifi in the library", "how do i renew a book", "which books are in the library", "where is the library", "library timings please", "how many books can i issue", "can i borrow a book",
                "library card details", "when does the library close", "is there a reading room", "how long can i keep a book"],
    "hostel": ["when do the hostel gates close", "how much is the hostel security deposit", "is laundry available in the hostel", "can girls get a hostel room", "who is the hostel warden", "hostel room allotment list", "is hostel available", "hostel fees for the year", "how do i apply for a hostel room", "mess timings in hostel",
               "can i get a single room", "hostel rules for visitors", "what are the hostel facilities", "hostel admission process"],
    "exam": ["what is the syllabus for the exam", "how many marks are there in the practical exam", "can i give a backlog exam", "when is the internal test", "what is the passing marks", "exam centre details", "when are the exams", "date sheet for end semester", "how is the practical exam conducted",
             "exam hall rules", "when will the results come", "re evaluation process", "admit card download", "viva schedule"],
    "placement": ["do companies come for core branches", "highest package of last year", "resume tips for placements", "is aptitude test conducted", "how many students got placed", "placement cell contact", "how are the placements", "which companies visit the campus", "placement training details",
                  "what is the average package", "when does placement season start", "eligibility for placements",
                  "internship opportunities", "campus recruitment drive"],
    "fees": ["what are the fee details", "fee structure for first year", "how much is the exam fee", "where to pay the fee", "scholarship application date", "caution money refund", "what is the semester fee", "last date to pay the fee", "can i pay fee in instalments", "fee payment link",
             "scholarship details", "fee refund policy", "late fee charges", "how to pay tuition fee online"],
    "greeting": ["hello assistant", "hey there", "good afternoon", "hi bot", "hello", "hi there", "good morning", "hey assistant", "good evening", "hello bot", "namaste", "hi"],
    "thanks": ["thanks assistant", "okay thank you", "many thanks", "thank you very much", "thank you", "thanks a lot", "that was helpful", "great thanks", "thank you so much", "thanks bye"],
    "other": ["what is the time now", "sing a song", "who won the world cup", "what is the price of petrol", "tell me a story", "what should i cook today", "what is the weather today", "tell me a joke", "who is the prime minister", "play a song",
              "what is two plus two", "how is the cricket score", "recommend a movie", "what is your favourite colour"],
}
replies = {
    "library": "The library is open from 9 am to 6 pm on weekdays. You can issue 3 books for 14 days.",
    "hostel": "Hostel forms are available at the office in July. Rooms are given on a first come basis.",
    "exam": "The date sheet is put on the notice board and the portal 15 days before the exams.",
    "placement": "The placement cell starts training in the sixth semester and drives begin in August.",
    "fees": "The semester fee must be paid by the 15th of the first month, online or at the accounts office.",
    "greeting": "Hello! I can help with the library, hostel, exams, placements and fees.",
    "thanks": "You are welcome! Ask anything else you need.",
    "other": "Sorry, I can only answer questions about college life.",
}
tags = list(faq)
base_texts = [s for t in tags for s in faq[t]]
base_labels = [i for i, t in enumerate(tags) for _ in faq[t]]


def variants(sentence, n=5):
    words = sentence.split()
    out = [sentence]
    for _ in range(n):
        if len(words) > 3:
            drop = random.randrange(len(words))
            out.append(" ".join(w for k, w in enumerate(words) if k != drop))
    return out


idx = np.random.default_rng(0).permutation(len(base_texts))
cut = int(0.8 * len(idx))
train_ids, val_ids = idx[:cut], idx[cut:]
texts, labels = [], []
for i in train_ids:
    for v in variants(base_texts[i]):
        texts.append(v)
        labels.append(base_labels[i])
labels = np.array(labels)
val_texts = [base_texts[i] for i in val_ids]
val_labels = np.array([base_labels[i] for i in val_ids])

vec = layers.TextVectorization(max_tokens=400, output_sequence_length=8)
vec.adapt(base_texts)
X = vec(np.array(texts)).numpy()
Xv = vec(np.array(val_texts)).numpy()
print(f"{len(base_texts)} example questions in {len(tags)} topics, vocabulary of {len(vec.get_vocabulary())} words")
print(f"training set after dropping one word at random: {len(texts)} sentences; held-out questions: {len(val_texts)}")

model = keras.Sequential([
    keras.layers.Input((8,)),
    layers.Embedding(400, 32, mask_zero=True),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.2),
    layers.Dense(len(tags), activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X, labels, validation_data=(Xv, val_labels), epochs=40, batch_size=32, verbose=0)
print(f"parameters {model.count_params()}, train accuracy {hist.history['accuracy'][-1] * 100:.1f} %, held-out accuracy {hist.history['val_accuracy'][-1] * 100:.1f} % (on {len(val_texts)} unseen questions)")
full_X = vec(np.array(base_texts)).numpy()
model.fit(full_X, np.array(base_labels), epochs=15, verbose=0)


def answer(msg):
    p = model.predict(vec(np.array([msg])).numpy(), verbose=0)[0]
    top = np.argsort(p)[::-1][:2]
    if p[top[0]] < 0.45:
        return "I am not sure what you mean. Try asking about the library, hostel, exams, placements or fees.", f"unsure ({tags[top[0]]} {p[top[0]]:.2f}, {tags[top[1]]} {p[top[1]]:.2f})"
    return replies[tags[top[0]]], f"{tags[top[0]]} {p[top[0]]:.2f}, next {tags[top[1]]} {p[top[1]]:.2f}"


print("\nchat session (type thanks to end)")
while True:
    msg = input("student> ")
    text, info = answer(msg)
    print(f"bot> {text}\n     [{info}]")
    if tags[int(np.argmax(model.predict(vec(np.array([msg])).numpy(), verbose=0)[0]))] == "thanks":
        break
```

## 🖥️ Output

```text
106 example questions in 8 topics, vocabulary of 185 words
training set after dropping one word at random: 339 sentences; held-out questions: 22
parameters 29960, train accuracy 100.0 %, held-out accuracy 77.3 % (on 22 unseen questions)

chat session (type thanks to end)
student> hi
bot> Hello! I can help with the library, hostel, exams, placements and fees.
     [greeting 1.00, next fees 0.00]
student> how many books can i take from the library
bot> The library is open from 9 am to 6 pm on weekdays. You can issue 3 books for 14 days.
     [library 0.96, next other 0.03]
student> when will the exam results be declared
bot> The date sheet is put on the notice board and the portal 15 days before the exams.
     [exam 1.00, next other 0.00]
student> which companies come for placements
bot> The placement cell starts training in the sixth semester and drives begin in August.
     [placement 1.00, next exam 0.00]
student> what is the capital of japan
bot> Sorry, I can only answer questions about college life.
     [other 0.99, next greeting 0.00]
student> thank you
bot> You are welcome! Ask anything else you need.
     [thanks 1.00, next hostel 0.00]
```

## ✅ Result

The data has 106 questions, which grow to 339 training sentences after augmentation, and the model has 29960 parameters. It fits the training sentences completely and answers 77.3 % of the 22 unseen questions correctly; in earlier runs without the augmentation the held-out accuracy was only 54 % to 68 %. In the chat the bot greeted correctly, answered the library question (0.96), the exam-results question (1.00) and the placement question (1.00), sent the off-topic question about Japan to the 'other' class, and ended on 'thank you'. With so little data it can still go wrong on unusual wording.

## 🎤 Viva Questions

**Q. Why a bi-directional LSTM?**  
A. It reads the sentence forwards and backwards, so each word is understood with the words before and after it.

**Q. What is an intent?**  
A. The goal behind the user's message, such as asking the lab time.

**Q. What is an embedding?**  
A. A learned vector of numbers that represents a word.

**Q. What does the softmax layer give?**  
A. The probability of each intent.

**Q. Why is a fallback class used?**  
A. So that off-topic questions get a polite refusal and not a wrong answer.

---
[⬅ Back to index](README.md)
