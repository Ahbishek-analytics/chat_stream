pip install transformers
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


user_input = st.text_input("Enter your message:")
if st.button("Submit"):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.success(output_text)


if __name__ == "__main__":
    st.title("Chat with GPT-2")
    st.write("I am a chatbot powered by GPT-2.")
    run_chatbot()
