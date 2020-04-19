import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras

from    PIL import Image
import  glob
from    gan import Generator, Discriminator

from    dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2) 
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    # loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,
    #                                                y_true=tf.ones_like(logits))
    return - tf.reduce_mean(logits)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    # loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,
    #                                                y_true=tf.zeros_like(logits))
    return tf.reduce_mean(logits)


def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) #[b]
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp



def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_real + d_loss_fake + 10. * gp

    return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):

    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)

    return loss


def main():

    tf.random.set_seed(233)
    np.random.seed(233)
    assert tf.__version__.startswith('2.')


    # hyper parameters
    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.0005
    is_training = True


    img_path = glob.glob(r'C:\Users\Jackie\Downloads\faces\*.jpg')
    assert len(img_path) > 0
    

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()
    db_iter = iter(dataset)


    generator = Generator() 
    generator.build(input_shape = (None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))
    z_sample = tf.random.normal([100, z_dim])


    g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)


    for epoch in range(epochs):

        for _ in range(5):
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)

            # train D
            with tf.GradientTape() as tape:
                d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        
        batch_z = tf.random.normal([batch_size, z_dim])

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss),
                  'gp:', float(gp))

            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'wgan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')



if __name__ == '__main__':
    main()