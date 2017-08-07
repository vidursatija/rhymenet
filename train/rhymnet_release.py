from __future__ import division
import tensorflow as tf
import numpy as np
from lyricreader2 import LyricsReader
import resource
import pickle
import os

"""
class BatchGenerator():

	def __init__(self, batch_size, num_steps, windows_size, vocab_size):
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.windows_size = windows_size
		self.vocab_size = vocab_size

		self.LR = LyricsReader(2015, 2016, 'lyrics_scraper/r-b-hip-hop-songs', vocab_size)
		print("Reading complete")
		self.n_files = self.LR.getSongsCount()

		self.fileNum = 0
		self.currentFilePointer = 0
		self.batchNum = 0
		self.current_file_lyrics = self.LR.lyricsAtIndex(0)
		self.current_file_rhymes = self.LR.rhymesAtIndex(0)

	def getNextBatch(self):
		thisBatchLyrics = np.empty((self.batch_size, self.num_steps), dtype=np.int16)
		thisBatchRyhmes = np.empty((self.batch_size, self.num_steps), dtype=np.int16)

		batches_filled = 0
		while batches_filled < self.batch_size:
			words_remain = len(self.current_file_lyrics) - self.currentFilePointer
			if words_remain < self.num_steps:
				self.currentFilePointer = 0
				self.fileNum = (self.fileNum + 1) % self.n_files
				self.current_file_lyrics = self.LR.lyricsAtIndex(self.fileNum)
				self.current_file_rhymes = self.LR.rhymesAtIndex(self.fileNum)
				#print("New file")
				continue

			batches_possible = int(np.ceil((words_remain-self.num_steps+1)/self.windows_size))
			if batches_possible > self.batch_size-batches_filled:
				words_required = (self.batch_size-batches_filled+2)*self.windows_size
				b = self.no_sum_conv(self.current_file_lyrics[self.currentFilePointer:self.currentFilePointer+words_required])
				b2 = self.no_sum_conv(self.current_file_rhymes[self.currentFilePointer:self.currentFilePointer+words_required])
				self.currentFilePointer += words_required
				#print(batches_possible)

				for i in range(len(b)):
					thisBatchLyrics[batches_filled] = b[i]
					thisBatchRyhmes[batches_filled] = b2[i]
					batches_filled += 1
				continue

			if batches_possible <= self.batch_size-batches_filled:
				words_required = (batches_possible+2)*self.windows_size
				b = self.no_sum_conv(self.current_file_lyrics[self.currentFilePointer:self.currentFilePointer+words_required])
				b2 = self.no_sum_conv(self.current_file_rhymes[self.currentFilePointer:self.currentFilePointer+words_required])
				self.currentFilePointer += words_required
				#print(batches_possible)

				for i in range(len(b)):
					thisBatchLyrics[batches_filled] = b[i]
					thisBatchRyhmes[batches_filled] = b2[i]
					batches_filled += 1
				continue

		return thisBatchLyrics, thisBatchRyhmes


	def no_sum_conv(self, all_words):
		b = []
		batches_possible = int(np.ceil((len(all_words)-self.num_steps+1)/self.windows_size))
		for i in range(batches_possible):
			b.append(all_words[i*self.windows_size:i*self.windows_size+self.num_steps])
		#print(len(b))
		return b

"""

hidden_size = 600
hidden_size_r = 50
vocab_size = 10000
vocab_size_r = 58
batch_size = 1
#num_steps = 31
#windows_size = 15
epoch = 2000
learn_rate = 0.2
avg_err = 0.0

#BG = BatchGenerator(batch_size, num_steps, windows_size, vocab_size)
LR = LyricsReader(2007, 2016, 'lyrics_scraper/r-b-hip-hop-songs', vocab_size)
n_files = LR.getSongsCount()
"""
sess = tf.InteractiveSession()

gru_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.5)#tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85), tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85)])#tf.contrib.rnn.GRUCell(hidden_size)
gru_cell_r = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size_r), output_keep_prob=0.3)
embeddings = tf.Variable(tf.truncated_normal([vocab_size, hidden_size], stddev=0.01, dtype=tf.float16), name='embedding')
embeddings_r = tf.Variable(tf.truncated_normal([vocab_size_r, hidden_size_r], stddev=0.01, dtype=tf.float16), name='embedding_r')

rev_embed = tf.Variable(tf.truncated_normal([hidden_size, vocab_size], stddev=0.01, dtype=tf.float16), name='rev_w')
rev_embed_r = tf.Variable(tf.truncated_normal([hidden_size_r, vocab_size], stddev=0.01, dtype=tf.float16), name='rev_w_r')
rev_bias = tf.Variable(tf.constant(0.05, shape=[vocab_size], dtype=tf.float16), name='rev_b')

inputXY = tf.placeholder(tf.int32, shape=[1, None])
input_y = tf.reshape(inputXY[:, 1:], [-1])
input_x = inputXY[:, :-1]
inputXY_r = tf.placeholder(tf.int32, shape=[1, None])
input_x_r = inputXY_r[:, :-1]

x = tf.nn.embedding_lookup(embeddings, input_x)
x = tf.nn.dropout(x, 0.5)
x_r = tf.nn.embedding_lookup(embeddings_r, input_x_r)
x_r = tf.nn.dropout(x_r, 0.3)

with tf.variable_scope("RNN"):
	outputs, f_state = tf.nn.dynamic_rnn(gru_cell, x, dtype=tf.float16)
with tf.variable_scope("RNN_r"):
	outputs_r, f_state_r = tf.nn.dynamic_rnn(gru_cell_r, x_r, dtype=tf.float16)

output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
output_r = tf.reshape(tf.concat(axis=1, values=outputs_r), [-1, hidden_size_r])
logits =  tf.matmul(output, rev_embed) + tf.matmul(output_r, rev_embed_r) + rev_bias
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [input_y], [tf.ones_like(input_y, dtype=tf.float16)])
cost = tf.reduce_sum(loss) / batch_size
learn_r = tf.Variable(learn_rate, trainable=False)

tvars = tf.trainable_variables()
optimizer = tf.train.GradientDescentOptimizer(learn_r)
gradsvars = optimizer.compute_gradients(cost)
#print(gradsvars)
grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 20)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
train_op = optimizer.apply_gradients(zip(grads, tvars))
new_lr = tf.placeholder(tf.float32, shape=[])
lr_update = tf.assign(learn_r, new_lr)
saver = tf.train.Saver()


sess.run(tf.global_variables_initializer())
"""
"""variables_names = [v.name for v in tvars]
sess.run(variables_names)
for k in variables_names:
	print(k)"""

for f in range(epoch):
	#sess.run(state_assign)
	#cost_eval, _ = sess.run([cost, train_op], feed_dict={inputXY: [LR.lyricsAtIndex(f%n_files)], inputXY_r: [LR.rhymesAtIndex(f%n_files)]})
	"""avg_err = (avg_err*(f) + cost_eval)/(f+1)
	if f%5 == 4:
		sess.run(lr_update, feed_dict={new_lr: learn_rate/(1+(0.002*f))})
		print ("Ep: "+str(f)+" avg:"+str(avg_err))

	if f%500 == 499:
		saver.save(sess, 'gru_rel', global_step=f)"""

	sL = len(LR.lyricsAtIndex(f%n_files))
	if sL < 35 or sL > 1500:
		print(sL)
		os.remove(LR.pathAtIndex(f%n_files))


#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
