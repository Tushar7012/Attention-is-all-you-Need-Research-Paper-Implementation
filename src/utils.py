import tensorflow as tf
import sacrebleu
import matplotlib.pyplot as plt


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def compute_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"Corpus BLEU: {bleu.score:.2f}")
    return bleu.score

def plot_attention(attention, sentence, predicted_sentence, layer):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    attention = attention[layer][0]
    ax.matshow(attention, cmap='viridis')
    ax.set_xticks(range(len(predicted_sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(predicted_sentence, rotation=90)
    ax.set_yticklabels(sentence)
    plt.show()
