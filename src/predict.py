import pickle
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "sentiment_rnn.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

MAX_LENGTH = 20

print("RNN Sentiment Analysis")
print("Type 'exit' to stop.")

while True:
    sentence = input("\nEnter a sentence: ").strip()

    if sentence.lower() == "exit":
        break

    if not sentence:
        print("Please enter a sentence.")
        continue

    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")

    probability = float(model.predict(padded, verbose=0)[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")
