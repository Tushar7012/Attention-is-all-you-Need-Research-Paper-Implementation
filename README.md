# 📚 Attention Is All You Need — Transformer (TensorFlow)

This project is a **full implementation** of the famous [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) paper (**Vaswani et al., 2017**) using **TensorFlow 2.x**, built **from scratch**.

It covers:
- Encoder & Decoder layers
- Multi-Head Scaled Dot-Product Attention
- Positional Encoding
- Custom Learning Rate Schedule (Noam)
- Teacher Forcing training loop
- BLEU score evaluation & inference

---

## 📌 **Dataset**

This demo uses **Portuguese → English** translation from the WMT dataset.  
Example test input:

---

## 📂 **Project Structure**
  Attention-Is-All-You-Need/
├── src/
│ ├── layers.py # Attention, FFN, Positional Encoding
│ ├── model.py # Full Transformer architecture
│ ├── utils.py # Masks, BLEU, helper functions
│ ├── dataset.py # Tokenizer, tf.data input pipeline
│ ├── train.py # Training loop (saves weights)
│ ├── inference.py # Loads weights & runs translation


---

## ⚙️ **Installation**

```bash
python -m venv venv
venv\Scripts\activate      

pip install -r requirements.txt

