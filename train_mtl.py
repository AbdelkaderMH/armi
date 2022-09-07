import argparse
import numpy as np
import torch
import torch.nn as nn

import preprocessing
import modeling
#import modeling2 as modeling
from barbar import Bar
import random
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report
import utils
import losses
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup,\
    get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#from pcgrad import PCGrad


def train(base_model, classifier,iterator, optimizer, cat_criterion,scheduler):

    # set the model in eval phase
    base_model.train(True)
    classifier.train(True)

    acc_category = 0
    acc_misogyny = 0
    loss_category = 0
    loss_misogyny = 0
    all_category_outputs = []
    all_category_labels = []
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
        category_target = label_input['category']

        # forward pass
        output =  base_model(**data_input)
        misogyny_logits, category_logits= classifier(output)



        category_probs = torch.softmax(category_logits, dim=1)


        misogyny_probs = torch.sigmoid(misogyny_logits)

        _, predicted_category = torch.max(category_probs, 1)
        predicted_misogyny = torch.round(misogyny_probs).squeeze()
        all_category_outputs.extend(predicted_category.squeeze().int().cpu().numpy())
        all_category_labels.extend(category_target.squeeze().int().cpu().numpy())
        all_misogyny_outputs.extend(predicted_misogyny.squeeze().int().cpu().numpy())
        all_misogyny_labels.extend(misogyny_target.squeeze().int().cpu().numpy())

        # compute the loss
        category_loss = cat_criterion(category_logits, category_target)
        misogyny_loss = F.binary_cross_entropy_with_logits(misogyny_logits.squeeze(), misogyny_target)


        total_loss =  misogyny_loss + category_loss

        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
        #torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        loss_category += category_loss.item()
        loss_misogyny += misogyny_loss.item()

        loss_total += total_loss.item()

    #all_category_outputs = utils.label_rule(all_misogyny_outputs, all_category_outputs)
    acc_category = accuracy_score(y_true=all_category_labels, y_pred=all_category_outputs)
    acc_misogyny = accuracy_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs)
    fscore_category = f1_score(y_true=all_category_labels, y_pred=all_category_outputs, average='macro')
    fscore_misogyny = f1_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs, average='macro')

    accuracies = {'Category': acc_category , 'F1_Category': fscore_category,
                          'F1_Misogyny': fscore_misogyny,'Misogyny': acc_misogyny }
    losses = {'Category': loss_category / len(iterator), 'Misogyny': loss_misogyny / len(iterator),
                      'Totalloss': loss_total / len(iterator)}
    return accuracies, losses

def evaluate(base_model, classifier, iterator, cat_criterion):
    # initialize every epoch
    acc_category = 0
    acc_misogyny = 0
    loss_category = 0
    loss_misogyny = 0
    all_category_outputs = []
    all_category_labels = []
    all_misogyny_outputs = []
    all_misogyny_labels = []

    loss_total= 0

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
            category_target = label_input['category']

            # forward pass
            output = base_model(**data_input)
            misogyny_logits, category_logits = classifier(output)

            category_probs = torch.softmax(category_logits, dim=1)

            misogyny_probs = torch.sigmoid(misogyny_logits)

            _, predicted_category = torch.max(category_probs, 1)
            predicted_misogyny = torch.round(misogyny_probs).squeeze()
            all_category_outputs.extend(predicted_category.squeeze().int().cpu().numpy())
            all_category_labels.extend(category_target.squeeze().int().cpu().numpy())
            all_misogyny_outputs.extend(predicted_misogyny.squeeze().int().cpu().numpy())
            all_misogyny_labels.extend(misogyny_target.squeeze().int().cpu().numpy())

            # compute the loss
            category_loss = cat_criterion(category_logits, category_target)
            misogyny_loss = F.binary_cross_entropy_with_logits(misogyny_logits.squeeze(), misogyny_target)

            total_loss = category_loss + misogyny_loss


            loss_category += category_loss.item()
            loss_misogyny += misogyny_loss.item()

            loss_total += total_loss.item()

            acc_category += utils.calc_accuracy(category_probs, category_target)
            acc_misogyny += utils.binary_accuracy(misogyny_probs, misogyny_target)
    #all_category_outputs = utils.label_rule(all_misogyny_outputs, all_category_outputs)
    acc_category = accuracy_score(y_true=all_category_labels, y_pred=all_category_outputs)
    acc_misogyny = accuracy_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs)
    fscore_category = f1_score(y_true=all_category_labels, y_pred=all_category_outputs, average='macro')
    fscore_misogyny = f1_score(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs, average='macro')
    print('*************** Misogyny Report*****************')
    print(classification_report(y_true=all_misogyny_labels, y_pred=all_misogyny_outputs, target_names=['none', 'misogyny'], digits=4))
    print('*************** Category Report*****************')
    print(classification_report(y_true=all_category_labels, y_pred=all_category_outputs,
                                target_names=['none','damning','derailing','discredit',
                                              'dominance', 'sexual harassment', 'stereotyping & objectification',
                                              'threat of violence'], digits=4))


    accuracies = {'Category': acc_category, 'F1_Category': fscore_category,
                          'F1_Misogyny': fscore_misogyny,'Misogyny': acc_misogyny }
    losses = {'Category': loss_category / len(iterator), 'Misogyny': loss_misogyny / len(iterator),
                      'Totalloss': loss_total / len(iterator)}
    return accuracies, losses

def train_full(config, train_loader, valid_loader):

    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    if config['cls'] =='1':
        classifier = modeling.MTCLSClassifier(in_feature=base_model.output_num()).to(device)
    elif config['cls'] =='2':
        classifier = modeling.MTATTClassifier(in_feature=base_model.output_num()).to(device)
    elif config['cls'] =='3':
        classifier = modeling.VHMTClassifier(in_feature=base_model.output_num(), nl= config['nl']).to(device)
    else:
        classifier = modeling.MTCLSClassifier(in_feature=base_model.output_num()).to(device)
        config['cls'] = '1'

    ## set optimizer and criterions
    #weights = torch.tensor([1, 2, 3, 1, 3, 4, 2, 3]).to(device)
    #weights = torch.tensor([1, 3061 / 669, 3061 / 105, 3061 / 2868, 3061 / 219, 3061 / 61, 3061 / 653, 3061 / 230]).to(device)
    #weight=weights
    cat_criterion = losses.FocalLoss(class_num=8)#losses.LabelSmoothingLoss(smoothing=0.2)#  focal_loss.FocalLoss(class_num=8)
    #cat_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)nn.CrossEntropyLoss().to(device)#
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(base_model.named_parameters())
    params_cls = list(classifier.named_parameters())
    params_loss = list(cat_criterion.named_parameters())
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
            ], #'lr': config['lr'] * 5,
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in params_cls if any(nd in n for nd in no_decay)
            ],#'lr': config['lr'] * 5,
            'weight_decay': 0.0
        },
#        {
#            'params': [
#                p for n, p in params_loss
#            ]
#        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"])
    #optimizer = optim.SGD(optimizer_grouped_parameters, lr=config["lr"])


    train_data_size = len(train_loader)
    num_train_steps = len(train_loader) * config['epochs']
    steps_per_epoch = int(train_data_size / config['batch_size'])
    warmup_steps = int(config['epochs'] * train_data_size * 0.1 / config['batch_size'])

    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    # Train model
    best_val_loss = float('+inf')
    best_cat_F1 = 0
    best_mis_F1 = 0
    best_cat_acc = 0
    best_mis_acc = 0

    epo = 0
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, classifier, train_loader, optimizer, cat_criterion, scheduler)
        valid_accuracies, valid_losses = evaluate(base_model, classifier,valid_loader, cat_criterion)


        val_loss = valid_losses['Totalloss']
        F1_cat = valid_accuracies['F1_Category']
        if F1_cat > best_cat_F1:

        #if epoch == 2:
        #if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_cat_F1 = valid_accuracies['F1_Category']
            best_mis_F1 = valid_accuracies['F1_Misogyny']
            best_cat_acc = valid_accuracies['Category']
            best_mis_acc = valid_accuracies['Misogyny']

            epo = epoch + 1
            print("save model's checkpoint")
            torch.save(base_model.state_dict(), f'./ckpts/best_basemodel_MTL_ArMI_+{config["lm"]}_{config["cls"]}_{str(config["seed"])}.pth')
            torch.save(classifier.state_dict(), f'./ckpts/best_classifier_MTL_ArMI_{config["lm"]}_{config["cls"]}_{str(config["seed"])}.pth')


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

    return best_cat_F1, best_mis_F1, best_cat_acc, best_mis_acc, best_val_loss
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
    parser.add_argument('--path', type=str, default='data/ArMI2021_training.tsv')


    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cls', type=str, default='2',
                        choices=['1', '2', '3', '4', '5'],
                        help='options: 1: MTCLSClassifier, 2: MTATTClassifier, 3: VHMTClassifier')
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

    f = open(f'results/MTL_{config["cls"]}_{args.lm_pretrained}_FL_{str(config["lr"])}_CE.txt', mode='w')

    #seed_all(12345)
    seeds = [12346] #, 12347, 12348, 12349, 12340, 12341, 12342, 12343, 12344]
    avg_f1_cat, avg_acc_cat, avg_f1_mis, avg_acc_mis = 0, 0, 0, 0
    for RANDOM_SEED in seeds:
        config['seed'] = RANDOM_SEED
        seed_all(RANDOM_SEED)

        train_loader, valid_loader = preprocessing.loadTrainValData(path=args.path, batchsize=args.batch_size, size=0.1, num_worker= 1, pretraine_path=config['pretrained_path'], seed=RANDOM_SEED)
        best_cat_F1, best_mis_F1, best_cat_acc, best_mis_acc, val_loss = train_full(config, train_loader, valid_loader)
        avg_f1_cat += best_cat_F1
        avg_acc_cat += best_cat_acc
        avg_f1_mis += best_mis_F1
        avg_acc_mis += best_mis_acc
        print(f'Val. F1 Category: {best_cat_F1 * 100:.2f}%  \t F1 Misogyny: {best_mis_F1 :.4f} \t Total_loss: {val_loss}')
        f.write(f' seed: {RANDOM_SEED} \t  best_cat_F1: {best_cat_F1} \t best_mis_F1: {best_mis_F1} \t best_cat_acc: {best_cat_acc} \t best_mis_acc: {best_mis_acc} \t val_loss: {val_loss} \n')
    f.write(f'avg_f1_mis : {avg_f1_mis / len(seeds)} \t avg_acc_mis: {avg_acc_mis / len(seeds)}\n')
    f.write(f'avg_f1_cat : {avg_f1_cat / len(seeds)} \t avg_acc_cat: {avg_acc_cat / len(seeds)}\n')
