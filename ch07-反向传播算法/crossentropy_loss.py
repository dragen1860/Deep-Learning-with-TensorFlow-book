import tensorflow as tf 


tf.random.set_seed(4323)

x=tf.random.normal([1,3])

w=tf.random.normal([3,2])

b=tf.random.normal([2])

y = tf.constant([0, 1])


with tf.GradientTape() as tape:

	tape.watch([w, b])
	logits = (x@w+b)
	loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))

grads = tape.gradient(loss, [w, b])
print('w grad:', grads[0])

print('b grad:', grads[1])