import os
import time
import math
import random
import argparse
import mxnet as mx
from dataset import load_dataset, make_vocab, tokenize, rnn_buckets, rnn_batches
from img2seq_model import FeatureExtractor, Seq2seqLSTM

def train(num_embed, num_hidden, num_layers, batch_size, image_size, caption_length, context, sgd=False):
    print("Loading dataset...", flush=True)
    training_set = load_dataset("data/training")
    vocab = make_vocab(training_set)
    vocab.save("model/vocabulary.json")
    training_set = tokenize(training_set, vocab)
    validating_set = load_dataset("data/validating")
    validating_set = tokenize(validating_set, vocab)

    features = FeatureExtractor(ctx=context)
    model = Seq2seqLSTM(vocab.size(), num_embed, num_hidden, num_layers)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    if os.path.isfile("model/img2seq_model.ckpt"):
        with open("model/img2seq_model.ckpt", "r") as f:
            ckpt_lines = f.readlines()
        ckpt_argv = ckpt_lines[-1].split()
        epoch = int(ckpt_argv[0])
        best_L = float(ckpt_argv[1])
        learning_rate = float(ckpt_argv[2])
        epochs_no_progress = int(ckpt_argv[3])
        model.load_parameters("model/img2seq_model.params", ctx=context)
    else:
        epoch = 0
        best_L = float("Inf")
        epochs_no_progress = 0
        learning_rate = 0.001
        model.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate)
    if sgd:
        print("Optimizer: SGD")
        trainer = mx.gluon.Trainer(model.collect_params(), "SGD",
                                   {"learning_rate": learning_rate, "momentum": 0.5, "clip_gradient": 5.0})
    else:
        print("Optimizer: Adam")
        trainer = mx.gluon.Trainer(model.collect_params(), "Adam",
                                   {"learning_rate": learning_rate, "clip_gradient": 5.0})
    print("Training...", flush=True)
    while learning_rate >= 1e-8:
        random.shuffle(training_set)
        ts = time.time()

        training_L = 0.0
        training_batch = 0
        for bucket, seq_len in rnn_buckets(training_set, [2 ** (i + 1) for i in range(int(math.log(caption_length, 2)))]):
            for image, target, label in rnn_batches(bucket, vocab, batch_size, image_size, seq_len, context):
                training_batch += 1
                source = features(image)
                hidden = model.begin_state(func=mx.nd.zeros, batch_size=source.shape[1], ctx=context)
                with mx.autograd.record():
                    output, hidden = model(source, target, hidden)
                    L = loss(output, label)
                    L.backward()
                trainer.step(source.shape[1])
                batch_L = mx.nd.mean(L).asscalar()
                if batch_L != batch_L:
                    raise ValueError()
                training_L += batch_L
                print("[Epoch %d  Bucket %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" %
                    (epoch, seq_len, training_batch, batch_L, training_L / training_batch, time.time() - ts), flush=True)

        validating_L = 0.0
        validating_batch = 0
        ppl = mx.metric.Perplexity(ignore_label=None)
        for bucket, seq_len in rnn_buckets(validating_set, [2 ** (i + 1) for i in range(int(math.log(caption_length, 2)))]):
            for image, target, label in rnn_batches(bucket, vocab, batch_size, image_size, seq_len, context):
                validating_batch += 1
                source = features(image)
                hidden = model.begin_state(func=mx.nd.zeros, batch_size=source.shape[1], ctx=context)
                output, hidden = model(source, target, hidden)
                L = loss(output, label)
                batch_L = mx.nd.mean(L).asscalar()
                if batch_L != batch_L:
                    raise ValueError()
                validating_L += batch_L
                probs = mx.nd.softmax(output, axis=1)
                ppl.update([label], [probs])

        epoch += 1

        avg_L = training_L / training_batch
        print("[Epoch %d]  learning_rate %.10f  training_loss %.10f  validating_loss %.10f  %s %f  epochs_no_progress %d  duration %.2fs" % (
            epoch, learning_rate, training_L / training_batch, validating_L / validating_batch, ppl.get()[0], ppl.get()[1], epochs_no_progress, time.time() - ts
        ), flush=True)

        if avg_L < best_L:
            best_L = avg_L
            epochs_no_progress = 0
            model.save_params("model/img2seq_model.params")
            with open("model/img2seq_model.ckpt", "a") as f:
                f.write("%d %.10f %.10f %d\n" % (epoch, best_L, learning_rate, epochs_no_progress))
        elif epochs_no_progress < 2:
            epochs_no_progress += 1
        else:
            epochs_no_progress = 0
            learning_rate *= 0.5
            trainer.set_learning_rate(learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a ai_challenger_caption trainer.")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    while True:
        try:
            train(
                num_embed = 128,
                num_hidden = 256,
                num_layers = 2,
                batch_size = 128,
                image_size = (224, 224),
                caption_length = 32,
                context = context,
                sgd = args.sgd
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
