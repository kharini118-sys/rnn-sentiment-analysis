# RNN-Based Sentiment Analysis of Customer Reviews

## Overview
This academic project implements a small sentiment-analysis application using a Simple Recurrent Neural Network (SimpleRNN). The model learns from labeled positive and negative reviews and predicts the sentiment of new sentences.

## Project Structure
- `data/reviews.csv` - small labeled review dataset
- `src/train.py` - preprocessing, model construction, training and evaluation
- `src/predict.py` - interactive prediction application
- `outputs/` - training plots and result files
- `requirements.txt` - required Python libraries
- `report/` - project report placeholder

## Method
1. Load labeled reviews.
2. Split data into training and testing sets.
3. Tokenize the training text.
4. Convert reviews to integer sequences.
5. Pad sequences to a fixed length.
6. Use an Embedding layer followed by SimpleRNN.
7. Use a sigmoid output for binary classification.
8. Evaluate the model.
9. Save the trained model and tokenizer.
10. Predict sentiment for new sentences.

## Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

The project is intentionally small so that the complete implementation can be understood and explained during viva.

## Authenticity / Acknowledgement
This project is prepared as an individual academic implementation. The dataset contains a small collection of manually written example reviews. Official TensorFlow/Keras and scikit-learn documentation may be consulted for API usage and conceptual understanding. Any external material used during further development should be acknowledged in the final submission.

## Git Practice
During actual development, make meaningful commits such as:
- Initialize project
- Add labeled review dataset
- Implement RNN training pipeline
- Add prediction application
- Add evaluation outputs
- Update documentation
