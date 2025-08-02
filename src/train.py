import tensorflow as tf
from model import Transformer
from utils import create_masks
from datasets import get_dataset
import time
import os


NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
EPOCHS = 10

train_dataset, tokenizer_pt, tokenizer_en = get_dataset()

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=1000,
    pe_target=1000,
    rate=DROPOUT_RATE
)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)   
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function(reduce_retracing=True)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)

output_dir = './outputs/checkpoints/'
os.makedirs(output_dir, exist_ok=True)

for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_state()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {train_loss.result():.4f}')

    print(f'Epoch {epoch+1} Loss {train_loss.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    print(f"✅ Finished Epoch {epoch+1}/{EPOCHS}")

transformer.save_weights(os.path.join(output_dir, 'transformer.weights.h5'))
print(f"✅ Weights saved to {output_dir}")
