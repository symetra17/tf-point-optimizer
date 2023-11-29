# Point position optimization
# in a random point pool,
# maximize inter point distance by tensorflow gradient decent
import numpy as np
import imageio
from tensorflow import keras
from tensorflow.keras import layers
from random import randint, shuffle
import tensorflow as tf
import cv2 as cv

batch_sz = 16
N = 48

o = keras.Sequential(
    [
        keras.layers.InputLayer((1,1,1)),
        layers.Conv2DTranspose( 2, (1, N) , use_bias=False),
    ],
    name="oil",
)



def loss_fn(y_true, y_pred):
    col = tf.reshape(y_pred, (batch_sz,N,1,2))
    row = tf.reshape(y_pred, (batch_sz,1,N,2))
    y = col - row
    y = tf.norm(y, ord='euclidean', axis=3,)
    y = tf.ragged.boolean_mask(y, y>0.)
    y = tf.reduce_min(y, axis=2)
    y = tf.reduce_mean(y)

    shifted = col - 200.
    r = tf.norm(shifted, ord='euclidean', axis=3)
    f = tf.math.square(r-100.) * tf.cast(r > 100., float)
    loss_b = .5*tf.reduce_mean(f)
    
    return -y + loss_b


opt = keras.optimizers.Adam(learning_rate=.2)
o.compile(optimizer=opt, loss=loss_fn)

#o.summary()

x = np.random.uniform(-1., 1., size=(1,1,N,2))
x = 30*x + 200
x = np.reshape(x, (1,1,N,2,1))
o.layers[0].set_weights(x)


xbatch = np.ones((batch_sz,1,1,1))
ybatch = np.random.uniform(0.0001, 1., size=(batch_sz,1,N,2))

for ntrain in range(100):
    o.fit( xbatch, ybatch, epochs=25 )
    w = o.layers[0].get_weights()[0]     # Shape   1,10,2,1    
    m = np.zeros((400,400,3), dtype=np.uint8)
    for n in range(w.shape[1]):
        p = w[0,n,:,0]
        p = tuple(p.astype(int))
        cv.circle(m, p, 7, (0,255,255), 2, cv.LINE_AA)
    cv.circle(m, (200,200), 100, (155,155,155), 2, cv.LINE_AA)
    cv.imwrite('result\\outp_%05d.png'%ntrain, m)

quit()


rand_list = []

for k in range(1):

    with tf.GradientTape() as tape:
        # Forward pass
        predic = o(xbatch)
        d_loss = loss_fn(ybatch, predic)
    # Calculate gradients with respect to every trainable variable
    grads = tape.gradient(d_loss, o.trainable_weights)


    o.optimizer.apply_gradients(zip(grads, o.trainable_weights))
