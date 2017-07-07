import time

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():

    #def fetch_data(self):
    #    print("in fetch")
    #    item = self.queue.dequeue()
    #    x = item["x"]
    #    y = item["y"]
    #        
    #    self.input_data = x.eval()
    #    self.targets = y.eval()


    def __init__(self, args, queue, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

              
        self.queue = queue


        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        #def fetch():
        #    print("in fetch")
        #    item = s.queue.dequeue()
        #    x = item["x"]
        #    y = item["y"]
        #    
        #    self.input_data = x
        #    self.targets = y
        #    return
        #    s.input_data = x
        #    s.targets = y
        #    
        #    s.embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])   
        #    x = tf.nn.embedding_lookup(s.embedding, x)
        #    print("got embedding")
        #    self.loaded_inputs = tf.split(x, args.seq_length, 1)
        #    self.loaded_inputs = [tf.squeeze(input_, [1]) for input_ in self.loaded_inputs]
        #    print("fetch done")            

        #self.fetch_data = fetch

        with tf.device("/cpu:0"):
            start_t = time.time()
            #item = self.queue.dequeue()
            #self.input_data = item["x"].eval()
            #self.targets = item["y"].eval()
            
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            print("processing took: ", time.time() - start_t)
            #tf.summary.scalar("input_process_time", time.time() - start_t)
        
            inputs = tf.split(inputs, args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            tf.summary.scalar("input_process_time", time.time() - start_t)

        # dropout beta testing: double check which one should affect next line
        #if training and args.output_keep_prob:
        #    inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        #inputs = tf.split(inputs, args.seq_length, 1)
        #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            tf.nn.embedding_lookup(self.embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        
        #with tf.device("/cpu:0"):
        #        embedding = tf.Variable(tf.stack(inputs, axis=0), trainable=False, name='embedding')

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)


        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        

        #loss = tf.nn.softmax_cross_entropy_with_logits(
        #        labels=tf.reshape(self.targets , [-1]),
        #        logits=self.logits,
        #        #tf.ones([args.batch_size * args.seq_length]))
        #)
        
        total_amnt = args.batch_size / args.seq_length
        
        #total_amnt = tf.Variable(total_amnt, "total_amnt", trainable=False)


        #self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / total_amnt

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        
        with tf.name_scope('optimizer'):
            train_rate = self.lr
            optimizer = tf.train.AdamOptimizer(train_rate)
        self.train_op = optimizer.minimize(loss) #apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1, correct=None):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        n_right = 0
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
         
            ret += pred
            char = pred
        
        #if correct is not None:
        #    equality = tf.equal(ret, prime +  correct)
        #    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        #    print("Equality: ", equality)
        #    print("Accuracy: ", accuracy)

        return ret

