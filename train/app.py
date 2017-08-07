from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
#from model import Model

class Model():

	def __init__(self, feed_forward=False, wi2pi=[], hidden_size=300, hidden_size_r=64, vocab_size=10000, vocab_size_r=37, learn_rate=0.1):

		self.hidden_size = hidden_size
		self.hidden_size_r = hidden_size_r
		self.vocab_size = vocab_size
		self.vocab_size_r = vocab_size_r
		self.hidden_units = hidden_size + hidden_size_r
		self.learn_rate = learn_rate

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units)

		if feed_forward == False:
			lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.75)#tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85), tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85)])#tf.contrib.rnn.GRUCell(hidden_size)

		rev_embed = tf.get_variable('rev_w', [self.hidden_units, vocab_size], dtype=tf.float16)
		#rev_embed_r = tf.Variable(tf.truncated_normal([hidden_size_r, vocab_size_r], stddev=0.01, dtype=tf.float16), name='rev_w_r')
		rev_bias = tf.get_variable('rev_b', [vocab_size], dtype=tf.float16)

		with tf.device("/cpu:0"):
			self.inputX = tf.placeholder(tf.int32, shape=[None])
			self.inputX_r = tf.placeholder(tf.int32, shape=[None])

			embeddings = tf.get_variable('embedding', [vocab_size, hidden_size], dtype=tf.float16)
			#embeddings_r = tf.get_variable('embedding_r', [vocab_size_r, hidden_size_r], dtype=tf.float16) #try one-hot

			x = tf.nn.embedding_lookup(embeddings, self.inputX)
			#x_r = tf.nn.embedding_lookup(embeddings_r, [self.inputX_r])
			x_r = tf.one_hot(self.inputX_r, self.hidden_size_r, dtype=tf.float16)
			if feed_forward == False:
				x = tf.nn.dropout(x, 0.75)
				x_r = tf.nn.dropout(x_r, 0.75)
				self.targets = tf.placeholder(tf.int32, shape=[None])
			x_i = tf.reshape(tf.concat([x, x_r], -1), [1, -1, self.hidden_units])

		with tf.variable_scope("RNN"):
			outputs, f_state = tf.nn.dynamic_rnn(lstm_cell, x_i, dtype=tf.float16)

		#Add tf while loop to control feed forward
		#vocab_size-2 is finish in word to id
		if feed_forward:
			self.feed_forward_outputs = [tf.argmax(tf.matmul(outputs[0][-1], rev_embed) + rev_bias, axis=0)]

			def body(state):
				#convert ffo to [1, 1, self.hidden_units]
				iX = tf.reshape(self.feed_forward_outputs[-1], [1, 1]) #number
				iX_r = tf.reshape(wi2pi[self.feed_forward_outputs[-1]], [1, 1])
				whole_x_i = tf.concat([tf.nn_embedding_lookup(embeddings, iX), tf.nn.embedding_lookup(embeddings_r, iX_r)], 2)
				op, new_state = lstm_cell(whole_x_i, state)
				y_0 = tf.argmax(tf.matmul(op[0], rev_embed) + rev_bias, axis=0)
				self.feed_forward_outputs.append(y_0)
				return new_state

			def condition(state):
				return tf.not_equal(feed_forward_outputs[-1], vocab_size-2)

			with tf.variable_scope("RNN"):
				final_state = tf.while_loop(condition, body, loop_vars=[f_state], shape_invariants=[f_state.get_shape()])

		else:
			output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_units])
			logits =  tf.matmul(output, rev_embed) + rev_bias
			loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [self.targets], [tf.ones_like(self.targets, dtype=tf.float16)])
			self.cost = tf.reduce_sum(loss)
			learn_r = tf.Variable(learn_rate, trainable=False)

			tvars = tf.trainable_variables()
			optimizer = tf.train.GradientDescentOptimizer(learn_r)
			gradsvars = optimizer.compute_gradients(self.cost)
			#print(gradsvars)
			grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 10)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
			self.train_op = optimizer.apply_gradients(zip(grads, tvars))
			self.new_lr = tf.placeholder(tf.float32, shape=[])
			self.lr_update = tf.assign(learn_r, self.new_lr)

		self.saver = tf.train.Saver()

	def run_n_epochs(self, sess, lyricsAtIndex, rhymesAtIndex, n_files, n=1):
		avg_err = 0.0
		for e in range(n):
			for f in range(n_files):
				cost_eval, _ = sess.run([self.cost, self.train_op], feed_dict={self.inputX: lyricsAtIndex[f][:-1], self.targets: lyricsAtIndex[f][1:], self.inputX_r: rhymesAtIndex[f][:-1]})
				avg_err = (avg_err*(f) + cost_eval)/(f+1)
				print(avg_err)
			sess.run(tf.Print(avg_err, [e, avg_err]))
			print(avg_err)

		sess.run(self.lr_update, feed_dict={self.new_lr: self.learn_rate/(1+(0.00075*n))})

		return avg_err

	def run_prediction(self, sess, start_lyrics, start_rhymes):
		f_outs = sess.run(self.feed_forward_outputs, feed_dict={self.inputX: start_lyrics, self.inputX_r: start_rhymes})
		return f_outs


tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("word_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_size_r", 37, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("vocab_size_r", 37, "French vocabulary size.")
tf.app.flags.DEFINE_string("rap_dir", "data/rap10k.pickle", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "end2end_rap", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("feed_forward", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_integer("n_steps", 50000, "Number of major epochs to run")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session()

	rap_file = tf.read_file(FLAGS.rap_dir)
	rap_file_data = pickle.loads(sess.run(rap_file))
	lyricsAtIndex = rap_file_data['all_lyrics']
	rhymesAtIndex = rap_file_data['all_rhymes']
	n_files = rap_file_data['count']
	word_id_to_phenome_id = rap_file_data['wi2pi']

	m = Model(feed_forward=False, wi2pi=word_id_to_phenome_id, hidden_size=FLAGS.word_size, hidden_size_r=FLAGS.word_size_r,
		vocab_size=FLAGS.vocab_size, vocab_size_r=FLAGS.vocab_size_r, learn_rate=FLAGS.learning_rate)

	sess.run(tf.global_variables_initializer())

	for i in range(int(FLAGS.n_steps*4/FLAGS.steps_per_checkpoint)):
		err = m.run_n_epochs(sess, lyricsAtIndex, rhymesAtIndex, n_files, n=int(FLAGS.steps_per_checkpoint/4))
		if i%4 == 3:
			m.saver.save(sess, FLAGS.model_dir)
		m.learn_rate = m.learn_rate*0.95
		#sess.run(tf.Print(err, [i, err]))

def predict():
	sess = tf.Session()

	rap_file = tf.read_file(FLAGS.rap_dir)
	rap_file_data = pickle.loads(sess.run(rap_file))
	word_to_id = rap_file_data['w2id']
	phenome_to_id = rap_file_data['p2id']
	word_id_to_phenome_id = rap_file_data['wi2pi']

	m = Model(feed_forward=True, wi2pi=word_id_to_phenome_id, hidden_size=FLAGS.word_size, hidden_size_r=FLAGS.word_size_r,
		vocab_size=FLAGS.vocab_size, vocab_size_r=FLAGS.vocab_size_r)

	m.saver.restore(sess, FLAGS.model_dir)

def main(_):
	if FLAGS.feed_forward == False:
		train()
	else:
		predict()

if __name__ == '__main__':
	tf.app.run()

		

