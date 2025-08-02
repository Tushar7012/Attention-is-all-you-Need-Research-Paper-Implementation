import tensorflow as tf
from utils import create_masks
from model import Transformer
from datasets import get_dataset   
import os

def evaluate(sentence, tokenizer_inp, tokenizer_tar, transformer, max_length=40):
    input_ids = [tokenizer_inp.vocab_size] + tokenizer_inp.encode(sentence) + [tokenizer_inp.vocab_size + 1]
    encoder_input = tf.expand_dims(input_ids, 0)


    output = [tokenizer_tar.vocab_size]
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input,
            output,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )

        predictions = predictions[:, -1:, :]  
        predicted_id = tf.argmax(predictions, axis=-1)

        predicted_id = tf.cast(predicted_id, tf.int32)

        if predicted_id == tokenizer_tar.vocab_size + 1:
            break

        output = tf.concat([output, predicted_id], axis=-1)

    predicted_sentence = tokenizer_tar.decode(
        [i for i in output[0].numpy() if i < tokenizer_tar.vocab_size]
    )
    return predicted_sentence, attention_weights

def translate(sentence, tokenizer_inp, tokenizer_tar, transformer):
    result, _ = evaluate(sentence, tokenizer_inp, tokenizer_tar, transformer)
    print(f'Input: {sentence}')
    print(f'Translation: {result}')
    return result

if __name__ == "__main__":
    _, tokenizer_pt, tokenizer_en = get_dataset()

    transformer = Transformer(
        num_layers=4,
        d_model=128,
        num_heads=8,
        dff=512,
        input_vocab_size=tokenizer_pt.vocab_size + 2,
        target_vocab_size=tokenizer_en.vocab_size + 2,
        pe_input=1000,
        pe_target=1000
    )

    dummy_encoder_input = tf.ones((1, 10), dtype=tf.int64)
    dummy_decoder_input = tf.ones((1, 10), dtype=tf.int64)
    transformer(
        dummy_encoder_input,
        dummy_decoder_input,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None
    )

    checkpoint_path = './outputs/checkpoints/transformer.weights.h5'
    transformer.load_weights(checkpoint_path)
    print(f" Loaded weights from {checkpoint_path}")

    translate("Meu nome é Tushar", tokenizer_pt, tokenizer_en, transformer)
