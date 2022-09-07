import argparse
import numpy as np
import torch
import torch.nn as nn

import preprocessing
import modeling
from barbar import Bar
import random
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score, classification_report
import utils
from transformers import AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#from pcgrad import PCGrad


def train(base_model, classifier,iterator, optimizer,scheduler):

    # set the model in eval phase
    base_model.train(True)
    classifier.train(True)

    acc_misogyny = 0
    loss_misogyny = 0
    all_misogyny_outputs = []
    all_misogyny_labels = []

    loss_total= 0


    for data_input, label_input  in Bar(iterator):

        for k, v in data_input.items():
            data_input[k] = v.to(device)

        for k, v in label_input.items():
            label_input[k] = v.to(device)

        optimizer.zero_grad()


        #forward pass

        misogyny_target = label_input['misogyny']

        # forward pass
        output =  base_model(**data_input)
        misogyny_logits = classifier(output)


        misogyny_probs = torch.sigmoid(misogyny_logits)

        predicted_misogyny = torch.round(misogyny_probs).squeeze()

        all_misogyny_outputs.extend(predicted_misogyny.squeeze().int().cpu().numpy())
        all_misogyny_labels.extend(misogyny_target.squeeze().int().cpu().numpy())

        # compute the loss
        misogyny_loss = F.binary_cross_entropy_with_logits(misogyny_logits.squeeze(), misogyny_target)

        misogyny_loss.backward()
        optimizer.step()
        scheduler.step()


        loss_misogyny += misogyny_loss.item()


    acc_misogyny = accuracy_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs)
    fscore_misogyny = f1_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs, average='macro')

    accuracies = {'F1_Misogyny': fscore_misogyny,'Misogyny': acc_misogyny}
    losses = {'Misogyny': loss_misogyny / len(iterator)}
    return accuracies, losses

def evaluate(base_model, classifier, iterator):
    # initialize every epoch
    acc_misogyny = 0
    loss_misogyny = 0
    all_misogyny_outputs = []
    all_misogyny_labels = []

    # set the model in eval phase
    base_model.eval()
    classifier.eval()

    with torch.no_grad():
        for data_input, label_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            for k, v in label_input.items():
                label_input[k] = v.to(device)

            misogyny_target = label_input['misogyny']

            # forward pass
            output = base_model(**data_input)
            misogyny_logits = classifier(output)

            misogyny_probs = torch.sigmoid(misogyny_logits)

            predicted_misogyny = torch.round(misogyny_probs).squeeze()
            all_misogyny_outputs.extend(predicted_misogyny.squeeze().int().cpu().numpy())
            all_misogyny_labels.extend(misogyny_target.squeeze().int().cpu().numpy())

            # compute the loss
            misogyny_loss = F.binary_cross_entropy_with_logits(misogyny_logits.squeeze(), misogyny_target)


            loss_misogyny += misogyny_loss.item()


    acc_misogyny = accuracy_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs)
    fscore_misogyny = f1_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs, average='macro')
    print('*************** Misogyny Report*****************')
    print(classification_report(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs, target_names=['none', 'misogyny'], digits=4))

    accuracies = {'F1_Misogyny': fscore_misogyny,'Misogyny': acc_misogyny }
    losses = { 'Misogyny': loss_misogyny / len(iterator)}
    return accuracies, losses

def train_full(config, train_loader, valid_loader):

    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    if config['cls'] =='1':
        classifier = modeling.CLSClassifier(in_feature=base_model.output_num(), class_num=1).to(device)
    elif config['cls'] =='2':
        classifier = modeling.ATTClassifier(in_feature=base_model.output_num(), class_num=1).to(device)
    elif config['cls'] =='3':
        classifier = modeling.VHClassifier(in_feature=base_model.output_num(), class_num=1).to(device)
    else:
        classifier = modeling.CLSClassifier(in_feature=base_model.output_num(), class_num=1).to(device)
        config['cls'] = '1'
    ## set optimizer and criterions
    cat_criterion = nn.CrossEntropyLoss().to(device)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(base_model.named_parameters())
    params_cls = list(classifier.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
        {
            'params': [
                p for n, p in params_cls
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in params_cls if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"])


    train_data_size = len(train_loader)
    num_train_steps = len(train_loader) * config['epochs']
    steps_per_epoch = int(train_data_size / config['batch_size'])
    warmup_steps = int(config['epochs'] * train_data_size * 0.1 / config['batch_size'])

    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # Train model
    best_val_loss = float('+inf')

    best_mis_F1 = 0
    best_mis_acc = 0

    epo = 0
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, classifier, train_loader, optimizer, scheduler)
        valid_accuracies, valid_losses = evaluate(base_model, classifier,valid_loader)

        val_loss = valid_losses['Misogyny']
        F1_mis = valid_accuracies['F1_Misogyny']

        if F1_mis > best_mis_F1:
        #if epoch ==2:
        #if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_mis_F1 = valid_accuracies['F1_Misogyny']
            best_mis_acc = valid_accuracies['Misogyny']

            epo = epoch + 1
            print("save model's checkpoint")
            torch.save(base_model.state_dict(), f'./ckpts/best_basemodel_MIS_ArMI_{config["lm"]}_{config["cls"]}_{str(config["seed"])}_.pth')
            torch.save(classifier.state_dict(), f'./ckpts/best_classifier_MIS_ArMI_{config["lm"]}_{config["cls"]}_{str(config["seed"])}.pth')


        print('********************Train Epoch***********************\n')
        print("accuracies**********")
        print("\t".join("{} : {}".format(k, v) for k, v in train_accuracies.items()))
        print("losses**********")
        print("\t".join("{} : {}".format(k, v) for k, v in train_losses.items()))
        print('********************Validation***********************\n')
        print("accuracies**********")
        print("\t".join("{} : {}".format(k, v) for k, v in valid_accuracies.items()))
        print("losses**********")
        print("\t".join("{} : {}".format(k, v) for k, v in valid_losses.items()))
        print('******************************************************\n')
    print(f"epoch of best results {epo}")

    return best_mis_F1,  best_mis_acc, best_val_loss
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
    parser.add_argument('--prepro', type=int, default=1)
    parser.add_argument('--path', type=str, default='data/ArMI2021_training.tsv')


    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cls', type=str, default='2',
                        choices=['1', '2', '3', '4', '5'],
                        help='options: 1: CLSClassifier, 2: ATTClassifier, 3: VHClassifier')
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
    config['cls'] = args.cls

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

    f = open(f'results/Misogyny_{config["cls"]}_{args.lm_pretrained}.txt', mode='w')

    #seed_all(12345)
    seeds = [12346]#, 12347, 12348, 12349, 12340, 12341, 12342, 12343, 12344]

    for RANDOM_SEED in seeds:
        config['seed'] = RANDOM_SEED
        seed_all(RANDOM_SEED)

        train_loader, valid_loader = preprocessing.loadTrainValData(path=args.path, batchsize=args.batch_size, size=0.1, num_worker= 1, pretraine_path=config['pretrained_path'], seed=RANDOM_SEED)
        best_mis_F1,  best_mis_acc, best_val_loss= train_full(config, train_loader, valid_loader)
        print(f'F1 Misogyny: {best_mis_F1 :.4f} \t Total_loss: {best_val_loss}')
        f.write(f' seed: {RANDOM_SEED}  \t best_mis_F1: {best_mis_F1} \t best_mis_acc: {best_mis_acc} \t val_loss: {best_val_loss} \n')

