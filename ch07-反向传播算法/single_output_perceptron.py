import tensorflow as tf 




x=tf.random.normal([1,3])

w=tf.ones([3,1])

b=tf.ones([1])

y = tf.constant([1])


with tf.GradientTape() as tape:

	tape.watch([w, b])
	logits = tf.sigmoid(x@w+b) 
	loss = tf.reduce_mean(tf.losses.MSE(y, logits))

grads = tape.gradient(loss, [w, b])
print('w grad:', grads[0])

print('b grad:', grads[1])


