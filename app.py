import streamlit as st
from pypinyin import pinyin, Style
from transformers import MarianMTModel, MarianTokenizer

# Function to convert Chinese lyrics to Hanyu Pinyin
def chinese_to_pinyin(text):
    pinyin_text = "\n".join(
        [" ".join([word[0] for word in pinyin(line, style=Style.NORMAL)]) for line in text.strip().split("\n")]
    )
    return pinyin_text

# Function to translate Chinese lyrics to English using a pre-trained model
def translate_chinese_to_english(text):
    # Load pre-trained MarianMT model and tokenizer for Chinese-to-English translation
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Perform translation
    translated_tokens = model.generate(**tokenized_text)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

# Streamlit app
st.title("Chinese Lyrics Translator")
st.write("Enter Chinese lyrics below to get Hanyu Pinyin and English translation.")

# Input text area for Chinese lyrics
chinese_lyrics = st.text_area("Paste Chinese Lyrics Here:", height=200)

if st.button("Translate"):
    if chinese_lyrics.strip():
        # Convert Chinese lyrics to Hanyu Pinyin
        pinyin_lyrics = chinese_to_pinyin(chinese_lyrics)

        # Translate Chinese lyrics to English
        english_translation = translate_chinese_to_english(chinese_lyrics)

        # Display results
        st.subheader("Hanyu Pinyin:")
        st.write(pinyin_lyrics)

        st.subheader("English Translation:")
        st.write(english_translation)
    else:
        st.warning("Please enter some Chinese lyrics.")