import os, re, json
import torch
import argparse
import pyhocon
import pickle
from nltk import tokenize


EOS_token = '<EOS>'
BOS_token = '<BOS>'
parallel_pattern = re.compile(r'^(.+?)(\t)(.+?)$')

# swbd_align = {
#     '<Uninterpretable>': ['%', 'x'],
#     '<Statement>': ['sd', 'sv', '^2', 'no', 't3', 't1', 'oo', 'cc', 'co', 'oo_co_cc'],
#     '<Question>': ['q', 'qy', 'qw', 'qy^d', 'bh', 'qo', 'qh', 'br', 'qrr', '^g', 'qw^d'],
#     '<Directive>': ['ad'],
#     '<Propose>': ['p'],
#     '<Greeting>': ['fp', 'fc'],
#     '<Apology>': ['fa', 'nn', 'ar', 'ng', 'nn^e', 'arp', 'nd', 'arp_nd'],
#     '<Agreement>': ['aa', 'aap', 'am', 'aap_am', 'ft'],
#     '<Understanding>': ['b', 'bf', 'ba', 'bk', 'na', 'ny', 'ny^e'],
#     '<Other>': ['o', 'fo', 'bc', 'by', 'fw', 'h', '^q', 'b^m', '^h', 'bd', 'fo_o_fw_"_by_bc'],
#     '<turn>': ['<turn>']
# }

damsl_align = {
    '<Uninterpretable>': ['abandoned_or_turn-exit/uninterpretable', 'non-verbal'],
    '<Statement>': ['statement-non-opinion', 'statement-opinion', 'collaborative_completion', 
        'other_answers', '3rd-party-talk', 'self-talk'],
    '<Question>': ['yes-no-question', 'wh-question', 'declarative_yes-no-question', 'backchannel_in_question_form',
        'open-question', 'rhetorical-questions', 'signal-non-understanding', 'or-clause', 'tag-question', 'declarative_wh-question'],
    '<Directive>': ['action-directive'],
    '<Propose>': ['offers,_options_commits'],
    '<Greeting>': ['conventional-opening', 'conventional-closing'],
    '<Apology>': ['apology', 'no_answers', 'reject', 'negative_non-no_answers', 'dispreferred_answers'],
    '<Agreement>': ['agree/accept', 'maybe/accept-part', 'thanking'],
    '<Understanding>': ['acknowledge_(backchannel)', 'summarize/reformulate', 'appreciation',
        'response_acknowledgement', 'affirmative_non-yes_answers', 'yes_answers'],
    '<Other>': ['other', 'quotation', 'repeat-phrase', 'hedge', 'hold_before_answer/agreement', 'downplayer'],
    '<turn>': ['<turn>']
}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', '-e', default='DAestimate', help='input experiment config')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    return args

def initialize_env(name):
    corpus_path = {
        'swda': {'path': './data/corpus/swda', 'pattern': r'^sw\_{}\_([0-9]*?)\.jsonlines$', 'lang': 'en'},
        'talkback': {'path': '/projects/anga5835/data/tb-jsonlines-filt', 'pattern': r'^{}\_(.*?)\.jsonlines$', 'lang': 'en'},
        'dailydialog': {'path': './data/corpus/dailydialog', 'pattern': r'^DailyDialog\_{}\_([0-9]*?)\.jsonlines$', 'lang': 'en'}
    }
    config = pyhocon.ConfigFactory.parse_file('experiments.conf')[name]
    config['log_dir'] = os.path.join(config['log_root'], name)
    config['train_path'] = corpus_path[config['corpus']]['path']
    config['corpus_pattern'] = corpus_path[config['corpus']]['pattern']
    config['lang'] = corpus_path[config['corpus']]['lang']
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    print('loading setting "{}"'.format(name))
    print('log_root: {}'.format(config['log_root']))
    print('corpus: {}'.format(config['corpus']))
    return config

class da_Vocab:
    def __init__(self, config, das=[], create_vocab=True):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.das = das
        if create_vocab:
            self.construct()
        else:
            self.load()

    def construct(self):
        vocab = {'<PAD>': 0, }
        vocab_count = {}
        for token in self.das:
            if token in vocab_count:
                vocab_count[token] += 1
            else:
                vocab_count[token] = 1
        for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
            vocab[k] = len(vocab)
        self.word2id = vocab
        self.id2word = {v : k for k, v in vocab.items()}
        return vocab

    def tokenize(self, X_tensor):
        X_tensor = [[self.word2id[token] for token in sentence] for sentence in X_tensor]
        return X_tensor

    def save(self):
        pickle.dump(self.word2id, open(os.path.join(self.config['log_root'], 'da_vocab.dict'), 'wb'))

    def load(self):
        self.word2id = pickle.load(open(os.path.join(self.config['log_root'], 'da_vocab.dict'), 'rb'))
        self.id2word = {v: k for k, v in self.word2id.items()}

class utt_Vocab:
    def __init__(self, config, sentences=[], create_vocab=True):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.sentences = sentences
        if create_vocab:
            self.construct()
        else:
            self.load()

    def construct(self):
        vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2, '<PAD>': 3, '<SEP>': 4}
        vocab_count = {}

        for sentence in self.sentences:
            for word in sentence:
                if word in vocab: continue
                if word in vocab_count:
                    vocab_count[word] += 1
                else:
                    vocab_count[word] = 1

        for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
            vocab[k] = len(vocab)
            if len(vocab) >= self.config['UTT_MAX_VOCAB']: break
        self.word2id = vocab
        self.id2word = {v : k for k, v in vocab.items()}
        return vocab

    def tokenize(self, X_tensor):
        X_tensor = [[[self.word2id[token] if token in self.word2id else self.word2id['<UNK>'] for token in seq] for seq in dialogue] for dialogue in X_tensor]
        return X_tensor

    def save(self):
        pickle.dump(self.word2id, open(os.path.join(self.config['log_root'], 'utterance_vocab.dict'), 'wb'))

    def load(self):
        self.word2id = pickle.load(open(os.path.join(self.config['log_root'], 'utterance_vocab.dict'), 'rb'))
        self.id2word = {v: k for k, v in self.word2id.items()}


def create_traindata(config, prefix='train'):
    file_pattern = re.compile(config['corpus_pattern'].format(prefix))
    files = [f for f in os.listdir(config['train_path']) if file_pattern.match(f)]
    da_posts = []
    da_cmnts = []
    utt_posts = []
    utt_cmnts = []
    turn = []
    # 1file 1conversation
    for filename in files:
        with open(os.path.join(config['train_path'], filename), 'r') as f:
            data = f.read().split('\n')
            data.remove('')
            da_seq = []
            utt_seq = []
            turn_seq = []
            # 1line 1turn
            for idx, line in enumerate(data, 1):
                jsondata = json.loads(line)
                for da, utt in zip(jsondata['DA'], jsondata['sentence']):
                    if config['lang'] == 'en':
                        _utt = [BOS_token] + en_preprocess(utt) + [EOS_token]
                    else:
                        _utt = [BOS_token] + utt.split(' ') + [EOS_token]
                    if config['corpus'] == 'swda':
                        da_seq.append(easy_damsl(da))
                    else:
                        da_seq.append(da)
                    utt_seq.append(_utt)
                    turn_seq.append(0)
                turn_seq[-1] = 1
            da_seq = [da for da in da_seq]
        if len(da_seq) <= config['window_size']: continue
        for i in range(max(1, len(da_seq) - 1 - config['window_size'])):
            assert len(da_seq[i:min(len(da_seq)-1, i + config['window_size'])]) >= config['window_size'], filename
            da_posts.append(da_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            da_cmnts.append(da_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
            utt_posts.append(utt_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            utt_cmnts.append(utt_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
            turn.append(turn_seq[i:min(len(da_seq), i + config['window_size'])])
    assert len(da_posts) == len(da_cmnts), 'Unexpect length da_posts and da_cmnts'
    assert len(utt_posts) == len(utt_cmnts), 'Unexpect length utt_posts and utt_cmnts'
    assert all(len(ele) == config['window_size'] for ele in da_posts), {len(ele) for ele in da_posts}
    return da_posts, da_cmnts, utt_posts, utt_cmnts, turn


def create_todbert_traindata(config, tokenizer, prefix='train'):
    file_pattern = re.compile(config['corpus_pattern'].format(prefix))
    files = [f for f in os.listdir(config['train_path']) if file_pattern.match(f)]
    da_posts = []
    da_cmnts = []
    utt_posts = []
    plain_utt_posts = []
    speaker_posts = []
    utt_cmnts = []
    turn = []
    # 1file 1conversation
    for filename in files:
        with open(os.path.join(config['train_path'], filename), 'r') as f:
            data = f.read().split('\n')
            data.remove('')
            da_seq = []
            utt_seq = []
            turn_seq = []
            plain_utt_seq = []
            speaker_seq = []
            # 1line 1turn
            for idx, line in enumerate(data, 1):
                jsondata = json.loads(line)
                speaker = jsondata['caller']
                if speaker == 'Teacher':
                    speaker_tok = '[SYS]'
                else:
                    speaker_tok = '[USR]'
                for da, utt in zip(jsondata['DA'], jsondata['sentence']):
                    plain_utt_seq.append(utt)
                    if config['lang'] == 'en':
                        _utt = [BOS_token] + en_preprocess(utt) + [EOS_token]
                    else:
                        _utt = [BOS_token] + utt.split(' ') + [EOS_token]
                    if config['corpus'] == 'swda':
                        da_seq.append(easy_damsl(da))
                    else:
                        da_seq.append(da)
                    utt_seq.append(_utt)
                    turn_seq.append(0)
                    speaker_seq.append(speaker_tok)
                turn_seq[-1] = 1
            da_seq = [da for da in da_seq]
        if len(da_seq) <= config['window_size']: continue
        for i in range(max(1, len(da_seq) - 1 - config['window_size'])):
            assert len(da_seq[i:min(len(da_seq)-1, i + config['window_size'])]) >= config['window_size'], filename
            da_posts.append(da_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            da_cmnts.append(da_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
            utt_posts.append(utt_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            plain_utt_posts.append(plain_utt_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            speaker_posts.append(speaker_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            utt_cmnts.append(utt_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
            turn.append(turn_seq[i:min(len(da_seq), i + config['window_size'])])
    assert len(da_posts) == len(da_cmnts), 'Unexpect length da_posts and da_cmnts'
    assert len(utt_posts) == len(utt_cmnts), 'Unexpect length utt_posts and utt_cmnts'
    assert all(len(ele) == config['window_size'] for ele in da_posts), {len(ele) for ele in da_posts}
    
    assert len(utt_posts) == len(plain_utt_posts), "Wrong tokenization"
    tod_context = []
    for i in range(len(plain_utt_posts)):
        context_str = "[CLS]"
        prev_speaker = None
        #assert len(tod_posts[i]) == len(plain_utt_posts[i]) == len(speaker_posts[i])
        assert len(plain_utt_posts[i]) == len(speaker_posts[i])
        for j in range(len(speaker_posts[i])):
            if speaker_posts[i][j] == prev_speaker:
                context_str = context_str + ' ' + plain_utt_posts[i][j]
            else:
                context_str = context_str + ' ' + speaker_posts[i][j] + ' ' + plain_utt_posts[i][j]
            prev_speaker = speaker_posts[i][j]
        #print(context_str)
        context_tokens = tokenizer.tokenize(context_str)
        context_tokenized = tokenizer.convert_tokens_to_ids(context_tokens)
        tod_context.append(context_tokenized)
    assert len(tod_context) == len(utt_posts)

    return da_posts, da_cmnts, utt_posts, tod_context, utt_cmnts, turn

def easy_damsl(tag):
    easy_tag = [k for k, v in damsl_align.items() if tag in v]
    return easy_tag[0] if not len(easy_tag) < 1 else tag

def separate_data(posts, cmnts, turn):
    split_size = round(len(posts) / 10)
    if split_size == 0: split_size = 1
    X_train, Y_train, Tturn = posts[split_size * 2:], cmnts[split_size * 2:], turn[split_size * 2:]
    X_valid, Y_valid, Vturn = posts[split_size: split_size * 2], cmnts[split_size: split_size * 2], turn[split_size: split_size * 2]
    X_test, Y_test, Testturn = posts[:split_size], cmnts[:split_size], turn[:split_size]
    assert len(X_train) == len(Y_train), 'Unexpect to separate train data'
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, Tturn, Vturn, Testturn

def en_preprocess(utterance):
    if utterance == '': return ['<Silence>']
    return tokenize.word_tokenize(utterance.lower())

