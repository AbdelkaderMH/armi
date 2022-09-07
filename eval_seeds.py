import argparse
import numpy as np
import torch
import torch.nn as nn

import preprocessing
import modeling
from barbar import Bar
import random
import pickle

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(base_model, classifier, iterator):
    all_sentiment_outputs = []

    # set the model in eval phase
    base_model.eval()
    classifier.eval()
    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            output, pooled = base_model(**data_input)
            sentiment_logits, out_neg, out_neu, out_pos = classifier(output, pooled)
            combined_logits = torch.cat([out_neg, out_neu, out_pos], dim=1)


            sentiment_probs = nn.Softmax(dim=1)(combined_logits).to(device)
            _, predicted_sentiment = torch.max(sentiment_probs, 1)
            all_sentiment_outputs.extend(predicted_sentiment.squeeze().int().cpu().numpy())

    return all_sentiment_outputs

def predict(base_model, classifier, iterator):
    all_sentiment_outputs = np.empty((0,3))
    all_sentiment_outputs_bin = np.empty((0,3))

    # set the model in eval phase
    base_model.eval()
    classifier.eval()
    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            output, pooled = base_model(**data_input)
            sentiment_logits, out_neg, out_neu, out_pos = classifier(output, pooled)
            combined_logits = torch.cat([out_neg, out_neu, out_pos], dim=1)

            sentiment_probs_1 = nn.Softmax(dim=1)(sentiment_logits).to(device)
            sentiment_probs_2 = nn.Softmax(dim=1)(combined_logits).to(device)

            #sentiment_probs = sentiment_probs_1 + sentiment_probs_2
            all_sentiment_outputs = np.vstack((all_sentiment_outputs, sentiment_probs_1.squeeze().float().cpu().numpy()))
            all_sentiment_outputs_bin = np.vstack((all_sentiment_outputs_bin, sentiment_probs_2.squeeze().float().cpu().numpy()))


    return all_sentiment_outputs, all_sentiment_outputs_bin


def eval_full(config, test_loader):
    seeds = [12346, 12347, 12348, 12349, 12340, 12341, 12342, 12343, 12344]
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True)
    classifier = modeling.MTClassifier2(class_num=3, in_feature=base_model.output_num())
    k= len(seeds)
    all_preds = []
    all_preds_bin = []
    for seed in seeds:
        base_model.load_state_dict(torch.load("./ckpts/best_basemodel_sentiment_"+config["lm"]+"_"+str(seed)+".pth"))
        classifier.load_state_dict(torch.load("./ckpts/best_classifier_sentiment_" +config["lm"]+"_"+str(seed)+".pth"))
        classifier.to(device)
        base_model.to(device)
        all_sentiment_outputs, all_sentiment_outputs_bin  = predict(base_model, classifier, test_loader)
        all_preds.append(all_sentiment_outputs)
        all_preds_bin.append(all_sentiment_outputs_bin)

    preds = (1 / (k) ) * np.sum(all_preds, axis=0)
    preds_bin = (1 / (k) ) * np.sum(all_preds_bin, axis=0)
    #sub = np.sum([preds, preds_bin], axis=0)  * 0.5
    #print(preds.shape)
    #df = pd.DataFrame(preds,columns=['neg', 'neu', 'pos'])
    #df.to_csv('results/last_sub_probs.csv', index=False, header=True)

    #df_bin = pd.DataFrame(preds_bin, columns=['neg', 'neu', 'pos'])
    #df_bin.to_csv('results/sub_probs_binary_bestsub794.csv', index=False, header=True)
    all_sentiment_outputs = np.argmax(preds_bin, axis=1)


    return all_sentiment_outputs, ids


if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--lm_pretrained', type=str, default='arabert',
                        help=" path of pretrained transformer")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_mult', type=float, default=1, help="dicriminator learning rate multiplier")

    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--prepro', type=int, default=12345)
    parser.add_argument('--path', type=str, default='data/test1_with_text.csv')


    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()


    config = {}
    config['args'] = args
    config["output_for_test"] = True
    config['epochs'] = args.epochs
    config["class_num"] = 1
    config["lr"] = args.lr
    config['lr_mult'] = args.lr_mult
    config['batch_size'] = args.batch_size
    config['lm'] = args.lm_pretrained

    if args.lm_pretrained == 'qarib':
        config['pretrained_path'] = "qarib/bert-base-qarib"
    elif args.lm_pretrained == 'camel':
        config['pretrained_path'] = "CAMeL-Lab/bert-base-camelbert-mix"
    elif args.lm_pretrained == 'arbert':
        config['pretrained_path'] = "UBC-NLP/ARBERT"
    elif args.lm_pretrained == 'marbert':
        config['pretrained_path'] = "UBC-NLP/MARBERT"
    elif args.lm_pretrained == 'larabert':
        config['pretrained_path'] = "aubmindlab/bert-large-arabertv02"
    else:
        config['pretrained_path'] = 'aubmindlab/bert-base-arabertv02'

    label_dict = {
        0: "-1",
        1: "0",
        2: "1"
    }

    test_loader, ids = preprocessing.loadTestData(path=args.path, batchsize=args.batch_size, num_worker=1, pretraine_path=config['pretrained_path'])
    all_sentiment, ids = eval_full(config, test_loader)

    submission = pd.DataFrame(columns=['Tweet_id','sentiment'])
    submission["sentiment"] = all_sentiment
    submission["sentiment"].replace(label_dict, inplace=True)
    submission["Tweet_id"] = ids
    submission.to_csv(f"results/test1_CSUM6P_submission_{config['lm']}_30.csv", index=False, header=True)

