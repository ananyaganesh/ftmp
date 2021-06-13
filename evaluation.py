from torch import optim
import time
import json
from utils import *
from models import *
from transformers import *
import random
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
import pandas as pd
from collections import Counter
from nltk import tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--expr', default='DAonly')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
def read_annotations():
    df = pd.read_csv('/projects/anga5835/data/diagnostic_test_set_gold.csv')
    full_das = []
    full_utts = []
    full_golds = []
    full_turns = []
    das = []
    utts = []
    golds = []
    turns = []
    speakers = []
    start_conv = True
    conv = 0
    for i, row in df.iterrows():
    
        utt_str = str(row['Utterance']).strip()
        if pd.isnull(row['ID']):
            continue
    
        if utt_str.startswith('Predicted'):
            full_das.append(das)
            full_utts.append(utts)
            full_golds.append(golds)
            full_turns.append(turns)
            das = []
            utts = []
            golds = []
            turns = []
            start_conv = True
            conv += 1
        else:
            split_line = utt_str.split(':')
            speaker = split_line[0]
            utt = ''.join(split_line[1:])
            utt_tok = ['<BOS>'] + tokenize.word_tokenize(utt.strip()) + ['<EOS>'] 
            if start_conv:
                turn = 1
            else:
                if speaker == speakers[-1]:
                    turn = 0
                else:
                    turn = 1
            speakers.append(speaker)
            utts.append(utt_tok)
            das.append(row['Talk move'])
            golds.append(row['Gold']) 
            turns.append(turn)
            start_conv = False
    return full_das, full_golds, full_utts, 0, full_turns


def evaluate(experiment):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    if config[use_tod]:
        XD_test, YD_test, XU_test, TC_test, _, turn_test = create_todbert_traindata(config=config, tokenizer=tokenizer, prefix='test')
    else:
        XD_test, YD_test, XU_test, _, turn_test = create_traindata(config=config, prefix='test')
    da_vocab = da_Vocab(config, create_vocab=False)
    utt_vocab = utt_Vocab(config, create_vocab=False)
    XD_test = da_vocab.tokenize(XD_test)
    YD_test = da_vocab.tokenize(YD_test)
    XU_test = utt_vocab.tokenize(XU_test)
    predictor = DApredictModel(utt_vocab=utt_vocab, da_vocab=da_vocab, tod_bert=tod_bert, config=config)
    predictor.load_state_dict(torch.load(os.path.join(config['log_dir'], 'da_pred_state8.model'), map_location=lambda storage, loc: storage))
    batch_size = config['BATCH_SIZE']
    k = 0
    indexes = [i for i in range(len(XU_test))]
    acc = []
    gold = []
    predicted = []
    macro_f = []
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        XU_seq = [XU_test[seq_idx] for seq_idx in batch_idx]
        XD_seq = [XD_test[seq_idx] for seq_idx in batch_idx]
        YD_seq = [YD_test[seq_idx] for seq_idx in batch_idx]
        turn_seq = [turn_test[seq_idx] for seq_idx in batch_idx]
        max_conv_len = max(len(s) for s in XU_seq)
        XU_tensor = []
        XD_tensor = []
        turn_tensor = []

        if config['use_tod']:
            TC_seq = [TC_test[seq_idx] for seq_idx in batch_idx]
            max_context_len = max(len(TC) for TC in TC_seq)
            for ci in range(len(TC_seq)):
                TC_seq[ci] = TC_seq[ci] + [0] * (max_context_len - len(TC_seq[ci]))
            TC_tensor = torch.tensor(TC_seq).to(device)
        else:
            TC_tensor = None
        for i in range(0, max_conv_len):
            max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
            for ci in range(len(XU_seq)):
                XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
            XU_tensor.append(torch.tensor([x[i] for x in XU_seq]).cpu())
            XD_tensor.append(torch.tensor([[x[i]] for x in XD_seq]).cpu())
            turn_tensor.append(torch.tensor([[t[i]] for t in turn_seq]).cpu())
        if config['DApred']['predict']:
            XD_tensor = XD_tensor[:-1]
            YD_tensor = torch.tensor([YD[-2] for YD in YD_seq]).cpu()
        else:
            YD_tensor = torch.tensor([YD[-1] for YD in YD_seq]).cpu()
        preds = predictor.predict(X_da=XD_tensor, X_utt=XU_tensor, TC=TC_tensor, turn=turn_tensor, step_size=step_size)
        preds = np.argmax(preds, axis=1)
        predicted.extend(preds)
        gold.extend(YD_tensor.data.tolist())
        acc.append(accuracy_score(y_pred=preds, y_true=YD_tensor.data.tolist()))
        macro_f.append(f1_score(y_true=YD_tensor.data.tolist(), y_pred=preds, average='macro'))
        k += step_size

    """
    with open('diagnostic_golds.json', 'w') as gf:
        json.dump(gold, gf)

    jsonable_preds = [int(x) for x in predicted]
    #print('Pred list type', len(predicted_list))
    with open('diagnostic_preds.json', 'w') as pf:
        json.dump(jsonable_preds, pf)
    """
    print(classification_report(gold, predicted, digits=4))
    print('Avg. Accuracy: ', np.mean(acc))
    print('Avg. macro-F: ', np.mean(macro_f))


if __name__ == '__main__':
    args = parse()
    evaluate(args.expr)
    
