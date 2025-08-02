import tensorflow as tf
import tensorflow_datasets as tfds

# -------------------------------
# ✅ Load & build tokenizers
# -------------------------------
def load_tokenizer():
    examples, metadata = tfds.load(
        'ted_hrlr_translate/pt_to_en',
        with_info=True,
        as_supervised=True
    )
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples),
        target_vocab_size=2**13
    )

    tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples),
        target_vocab_size=2**13
    )

    return train_examples, val_examples, tokenizer_pt, tokenizer_en

# -------------------------------
# ✅ Encode single example
# -------------------------------
def encode(pt, en, tokenizer_pt, tokenizer_en):
    pt = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(pt.numpy()) + [tokenizer_pt.vocab_size + 1]
    en = [tokenizer_en.vocab_size] + tokenizer_en.encode(en.numpy()) + [tokenizer_en.vocab_size + 1]
    return pt, en

# -------------------------------
# ✅ tf_encode closure (correct!)
# -------------------------------
def get_tf_encode(tokenizer_pt, tokenizer_en):
    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(
            func=lambda pt, en: encode(pt, en, tokenizer_pt, tokenizer_en),
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
    return tf_encode

# -------------------------------
# ✅ Filter by max length
# -------------------------------
def filter_max_length(x, y, max_length=40):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

# -------------------------------
# ✅ Final dataset loader
# -------------------------------
def get_dataset(BUFFER_SIZE=20000, BATCH_SIZE=64, MAX_LENGTH=40):
    train_examples, val_examples, tokenizer_pt, tokenizer_en = load_tokenizer()

    tf_encode = get_tf_encode(tokenizer_pt, tokenizer_en)

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(
        lambda x, y: filter_max_length(x, y, MAX_LENGTH)
    )
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([None], [None])
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, tokenizer_pt, tokenizer_en
