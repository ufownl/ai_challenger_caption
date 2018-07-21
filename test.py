import math
import argparse
import mxnet as mx
from vocab import Vocabulary
from dataset import load_image, cook_image
from img2seq_model import FeatureExtractor, Seq2seqLSTM

num_embed = 128
num_hidden = 1024
num_layers = 2
image_size = (224, 224)
beam_size = 10

parser = argparse.ArgumentParser(description="Start a ai_challenger_caption tester.")
parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)

print("Loading vocabulary...", flush=True)
vocab = Vocabulary()
vocab.load("model/vocabulary.json")

print("Loading model...", flush=True)
features = FeatureExtractor(ctx=context)
model = Seq2seqLSTM(vocab.size(), num_embed, num_hidden, num_layers)
model.load_parameters("model/img2seq_model.params", ctx=context)

for path in args.images:
    print(path)
    image = cook_image(load_image(path), image_size)
    image = image.T.expand_dims(0).as_in_context(context)
    source = features(image)
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
    enc_out, hidden = model.encode(source, hidden)
    target = mx.nd.array([vocab.word2idx("<GO>")], ctx=context)
    sequences = [([vocab.word2idx("<GO>")], 0.0, hidden)]
    while True:
        candidates = []
        for seq, score, hidden in sequences:
            if seq[-1] == vocab.word2idx("<EOS>"):
                candidates.append((seq, score, hidden))
            else:
                target = mx.nd.array([seq[-1]], ctx=context)
                output, hidden = model.decode(target.reshape((1, -1)).T, hidden, enc_out)
                probs = mx.nd.softmax(output, axis=1)
                beam = probs.reshape((-1,)).topk(k=beam_size, ret_typ="both")
                for i in range(beam_size):
                    candidates.append((seq + [int(beam[1][i].asscalar())], score + math.log(beam[0][i].asscalar()), hidden))
        if len(candidates) <= len(sequences):
            break;
        sequences = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

    scores = mx.nd.array([score for _, score, _ in sequences], ctx=context)
    probs = mx.nd.softmax(scores)

    for i, (seq, score, _) in enumerate(sequences):
        text = ""
        for token in seq[1:-1]:
            text += vocab.idx2word(token)
        print(text, score, probs[i].asscalar())
