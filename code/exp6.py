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
