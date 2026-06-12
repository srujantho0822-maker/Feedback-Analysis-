import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import joblib

# ------------------------------
# Step 1: Load the Excel file
# ------------------------------
file_path = "finalDataset0.2.xlsx"  
df = pd.read_excel(r"D:\feedback project\finalDataset0.2.xlsx")

# ------------------------------
# Step 2: Fix column names
# ------------------------------
df.columns = [c.strip() for c in df.columns]  # remove leading/trailing spaces
df = df.rename(columns={
    'coursecontent': 'coursecontent_text',
    'coursecontent_com': 'coursecontent_score',
    'examination': 'examination_text',
    'examination_com': 'examination_score',
    'labwork': 'labwork_text',
    'labwork_com': 'labwork_score',
    'library_facilities': 'library_text',
    'library_facilities_com': 'library_score',
    'extracurricular': 'extracurricular_text',
    'extracurricular_com': 'extracurricular_score',
    'problem_solving': 'problem_solving_score',
    'overall_teaching': 'overall_score',
    'teach_com': 'teaching_score',
})

# ------------------------------
# Step 3: Prepare text and numeric features
# ------------------------------
text_columns = [
    'teaching', 'coursecontent_text', 'examination_text',
    'labwork_text', 'library_text', 'extracurricular_text'
]

numeric_columns = [
    'teaching_score', 'coursecontent_score', 'examination_score',
    'labwork_score', 'library_score', 'extracurricular_score', 'problem_solving_score'
]
for col in numeric_columns:
    # Convert to numeric, coerce errors to NaN, then fill NaN with 0.
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

# Now X_numeric is safe 
X_numeric = df[numeric_columns].values

# Fill NaN and convert text to string 
for col in text_columns:
    df[col] = df[col].astype(str).fillna('')

# Fill NaN for numeric columns
for col in numeric_columns:
    df[col] = df[col].fillna(0)

# Combine text columns
df['all_text'] = df[text_columns].agg(' '.join, axis=1)

X_numeric = df[numeric_columns].values
labels = df['overall_score'].fillna(0).astype(int)  # target

# ------------------------------
# Step 4: Tokenize text
# ------------------------------
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['all_text'])
X_text = tokenizer.texts_to_sequences(df['all_text'])
X_text = pad_sequences(X_text, maxlen=max_len, padding='post')

# Save tokenizer
joblib.dump(tokenizer, 'tokenizer.pkl')

# ------------------------------
# Step 5: Split train/test
# ------------------------------
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_numeric, labels, test_size=0.2, random_state=40
)

# ------------------------------
# Step 6: Build model with text + numeric input
# ------------------------------
# Text input branch
text_input = Input(shape=(max_len,), name='text_input')
x = Embedding(max_words, 128)(text_input)
x = SpatialDropout1D(0.2)(x)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)

# Numeric input branch
num_input = Input(shape=(X_numeric.shape[1],), name='numeric_input')
y = Dense(64, activation='relu')(num_input)
y = Dropout(0.2)(y)

# Combine branches
combined = Concatenate()([x, y])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.2)(z)
output = Dense(1, activation='linear')(z)  # regression

model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ------------------------------
# Step 7: Train model
# ------------------------------
history = model.fit(
    [X_train_text, X_train_num], y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# ------------------------------
# Step 8: Evaluate and save
# ------------------------------
loss, mae = model.evaluate([X_test_text, X_test_num], y_test, verbose=1)
print(f"Test MAE: {mae}")

model.save("feedback_model_multimodal.h5")
print("Model saved as feedback_model_multimodal.h5")
