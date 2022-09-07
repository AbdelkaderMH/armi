import argparse
import numpy as np
import torch
import utils
import preprocessing
import modeling
from barbar import Bar
import random
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dict_misogyny = {
    0:'none' ,
    1: 'misogyny'
}
dict_category = {
    0: 'none',
    1: 'damning',
    2: 'derailing',
    3: 'discredit',
    4: 'dominance',
    5:'sexual harassment',
    6: 'stereotyping & objectification',
    7: 'threat of violence'
}


def evaluate(base_model, classifier, iterator):

    all_category_outputs = []

    all_misogyny_outputs = []

    # set the model in eval phase
    base_model.eval()
    classifier.eval()

    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)


            # forward pass
            output = base_model(**data_input)
            misogyny_logits, category_logits = classifier(output)

            category_probs = torch.softmax(category_logits, dim=1)

            misogyny_probs = torch.sigmoid(misogyny_logits)

            _, predicted_category = torch.max(category_probs, 1)
            predicted_misogyny = torch.round(misogyny_probs).squeeze()
            all_category_outputs.extend(predicted_category.squeeze().int().cpu().numpy())
            all_misogyny_outputs.extend(predicted_misogyny.squeeze().int().cpu().numpy())

    all_category_outputs = utils.label_rule(all_misogyny_outputs, all_category_outputs)



    return all_misogyny_outputs, all_category_outputs

def eval_full(config, valid_loader):

    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True)
    if config['cls'] =='1':
        classifier = modeling.MTCLSClassifier(in_feature=base_model.output_num()).to(device)
    elif config['cls'] =='2':
        classifier = modeling.MTATTClassifier(in_feature=base_model.output_num()).to(device)
    elif config['cls'] =='3':
        classifier = modeling.VHMTClassifier(in_feature=base_model.output_num(), nl= config['nl']).to(device)
    else:
        classifier = modeling.MTCLSClassifier(in_feature=base_model.output_num()).to(device)
        config['cls'] = '1'


    base_model.load_state_dict(torch.load(f'./ckpts/best_basemodel_MTL_ArMI_+{config["lm"]}_{config["cls"]}_{str(config["seed"])}.pth'))
    classifier.load_state_dict(torch.load(f'./ckpts/best_classifier_MTL_ArMI_{config["lm"]}_{config["cls"]}_{str(config["seed"])}.pth'))
    classifier.to(device)
    base_model.to(device)

    all_misogyny_outputs, all_category_outputs = evaluate(base_model, classifier, valid_loader)


    return all_misogyny_outputs, all_category_outputs

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    parser.add_argument('--lr_mult', type=float, default=2, help="dicriminator learning rate multiplier")

    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--nl', type=int, default=6)
    parser.add_argument('--path', type=str, default='data/ArMI2021_test.tsv')


    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cls', type=str, default='1',
                        choices=['1', '2', '3', '4', '5'],
                        help='options: 1: MTCLSClassifier, 2: MTATTClassifier, 3: VHMTClassifier')
    parser.add_argument('--seed', type=int, default=12346,
                        choices=[12345, 12346, 12347, 12348, 12349, 12340, 12341, 12342, 12343, 12344],
                        help='random seed must be in 12345, 12346, 12347, 12348, 12349, 12340, 12341, 12342, 12343, 12344 ')
    args = parser.parse_args()


    config = {}
    config['args'] = args
    config["output_for_test"] = True
    config['epochs'] = args.epochs
    config["lr"] = args.lr
    config['lr_mult'] = args.lr_mult
    config['batch_size'] = args.batch_size
    config['lm'] = args.lm_pretrained
    config['cls'] = args.cls
    config['seed'] = args.seed
    config['nl'] = args.nl

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

    seed_all(args.seed)

    valid_loader, ids = preprocessing.loadTestData(path=args.path, batchsize=args.batch_size, num_worker= 1, pretraine_path=config['pretrained_path'])
    all_misogyny_outputs, all_category_outputs = eval_full(config, valid_loader)

    df_sub = pd.DataFrame(columns=['tweet_id', 'misogyny', 'category'])
    df_sub['tweet_id'] = ids
    df_sub['misogyny'] = all_misogyny_outputs
    df_sub['misogyny'].replace(dict_misogyny, inplace=True)
    df_sub['category'] = all_category_outputs
    df_sub['category'].replace(dict_category, inplace=True)

    df_sub.to_csv('results/CS-UM6P_run2.csv', index=False, header=True, sep='\t')
