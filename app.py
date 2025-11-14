import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------
# LOAD YOUR FINE-TUNED MODEL
# ---------------------------
MODEL_PATH = "./outputs_llama31_8b_hscode"   # change if needed

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

st.title("HS Code Llama 3.1 — Test Chat")

# ---------------------------
# TEXT INPUT
# ---------------------------
user_input = st.text_input("Ask something:")

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Enter a question.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.4,
                top_p=0.9
            )

        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Only show answer (remove prompt)
        clean_answer = output_text[len(user_input):].strip()

        st.subheader("Model Response:")
        st.write(clean_answer)
