from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)




#word_chunk = "Given a monocular image, our aim is to localize the objects  in 3D by enclosing them with tight  oriented 3D bounding boxes. We propose a novel approach that extends the well-acclaimed deformable part-based model[Felz.]"

word_chunk = "\\noindent For any scheme $X$ the category $\\QCoh(\\mathcal{O}_X)$ of quasi-coherent modules is abelian and a weak Serre subcategory of the abelian category ofll $\\mathcal{O}_X$-modules. The same thing works for the category of quasi-coherent modules on an algebraic space $X$ viewed as a subcategory of the category of all $\\mathcal{O}_X$-modules on the small \\'etale site of $X$."


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\n\n\n")
            ##print(chars)
            ##print("\n\nSome generated text: ")

            primer = word_chunk[:6]
            correct = word_chunk[6:]

            print("using primer: ", primer)

            batches = list()
            i = 0
            chunk_size = 10
            for x in range(len(word_chunk) // chunk_size):
                batchno = x
                chunk = word_chunk[i:i+chunk_size]
                batches.append({
                    "primer": chunk[:chunk_size // 2],
                    "correct": chunk
                })
                i += chunk_size

            for batch in batches:
                correct = batch["correct"]
                primer = batch["primer"]

                txt = model.sample(sess, chars, vocab, len(correct),
                        primer, args.sample, correct)
                print("\nExpected:\n", correct)
                print("\nGot:\n", txt)
                
                
                equality = tf.equal(txt, correct[:chunk_size//2])
                accuracy = tf.reduce_mean(tf.cast(equality, tf.int32))
                accuracy = tf.reduce_sum(accuracy)
                accuracy = accuracy / len(txt)
                print("Equality: ", equality)
                print("Accuracy: ", accuracy)


            #txt = model.sample(sess, chars, vocab, args.n, primer,
            #        args.sample, correct=correct)

            #print("Expected:\n")
            #print(word_chunk)

            #print("Got:\n")
            #print(txt)


if __name__ == '__main__':
    main()
