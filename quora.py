import pandas as pd
import numpy as np
import re
import keras.backend as K
from keras.layers import *
from keras.models import *
from torchtext.data import Field, Dataset, Example, BucketIterator
from nltk.tokenize import wordpunct_tokenize
from itertools import chain, islice
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from unidecode import unidecode
from operator import itemgetter
import matplotlib.pyplot as plt

K.tensorflow_backend._get_available_gpus()

def _depth(obj):
    """Helper function to determine the depth of a nested list structure."""
    return isinstance(obj, (list, tuple)) and max(map(_depth, obj)) + 1


class WrapIterator(object):
    """
    Wraps a `torchtext.data.Iterator` to be used as data generator with Keras.
    Arguments:
        iterator: `torchtext.data.Iterator` instance to be wrapped.
        x_fields: Can be used to specify which field names correspond to input data. If None, Field names with
        is_target attribute set to False will be considered as input fields.
        y_fields: Can be used to specify which field names correspond to target data. If None, Field names with
        is_target attribute set to True will be considered as target fields.
        permute: Either None or a dictionary where each key is a field name that points to a list of dimension indices
        by which the corresponding output tensors should be permuted.
    Example:
        >>> dataset = Dataset(examples, [('text', text), ('label', label)])
        >>> iterator = Iterator(dataset, batch_size=32)
        >>> data_gen = WrapIterator(iterator)
        >>> model.fit_generator(iter(data_gen), steps_per_epoch=len(data_gen))
    """
	def __init__(self, iterator, x_fields=None, y_fields=None, permute=None):
        self.iterator = iterator
        self.permute = permute
        self.x_fields = []
        self.y_fields = []

        self.x_fields = self._process_fields_argument(x_fields, False)
        self.y_fields = self._process_fields_argument(y_fields, True)

    def _process_fields_argument(self, field_names, is_target):
        result = []
        if field_names is None:
            for name, field in self.iterator.dataset.fields.items():
                if issubclass(field.__class__, Field):
                    if not hasattr(field, 'is_target'):
                        raise Exception('Field instance does not have a is_target attribute and input and output '
                                        'fields also haven\'t been provided specifically.'
                                        ' Consider to upgrade torchtext or provide all input and output fields '
                                        'with the x_fields and y_fields arguments.')
                    if field.is_target == is_target:
                        result.append(name)
        else:
            all_field_names = list(self.iterator.dataset.fields.keys())
            for name in field_names:
                if name not in all_field_names:
                    raise ValueError('Provided input field \'{}\' is not in dataset\'s field list'.format(name))
            result.extend(field_names)

        if not result:
            raise Exception('No {} fields have been provided. Either provide fields with is_target attribute set to '
                            '{} or pass a list of field names as the {} argument'
                            .format('target' if is_target else 'input',
                                    is_target,
                                    'y_fields' if is_target else 'x_fields'))
        return result

    @classmethod
    def wraps(cls, iterators, x_fields=None, y_fields=None, **kwargs):
        """
        Wrap multiple iterators.
        Arguments:
            iterators: List of iterators to wrap.
            x_fields: Can be used to specify which field names correspond to input data. If None, Field names with
            is_target attribute set to False will be considered as input fields.
            y_fields: Can be used to specify which field names correspond to target data. If None, Field names with
            is_target attribute set to True will be considered as target fields.
            **kwargs: Arguments that will be passed to the constructor of `WrapIterator` instances.
        Example:
            >>> splits = Dataset.splits()
            >>> iterators = Iterator.splits(splits, batch_size=32)
            >>> train, test = WrapIterator(iterators)
            >>> model.fit_generator(iter(train), steps_per_epoch=len(train))
            >>> model.evaluate_generator(iter(test), steps=len(test))
        """
        def process_fields(fields, name):
            if fields is not None:
                depth = _depth(fields)
                if depth not in [1, 2]:
                    raise ValueError('\'{}\' must be either a list of field names or a list of'
                                     ' field name lists, one for each iterator'.format(name))
                fields = fields if depth == 2 else [fields] * len(iterators)
            return fields

        x_fields = process_fields(x_fields, 'x_fields')
        y_fields = process_fields(y_fields, 'y_fields')

        wrappers = []
        for i, it in enumerate(iterators):
            x_fields_arg = x_fields[i] if x_fields else None
            y_fields_arg = y_fields[i] if y_fields else None
            wrappers.append(cls(it, x_fields_arg, y_fields_arg, **kwargs))
        return wrappers

    def _process(self, tensor, field_name):
        if self.permute and field_name in self.permute:
            tensor = tensor.permute(*self.permute[field_name])
        return tensor.cpu().numpy()

    def __iter__(self):
        for batch in iter(self.iterator):
            batch_x = [self._process(getattr(batch, field), field) for field in self.x_fields]
            batch_y = [self._process(getattr(batch, field), field) for field in self.y_fields]
            yield batch_x, batch_y

    def __len__(self):
        return len(self.iterator)
		
	def balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score
	
	spaces_re = re.compile(r'\s+')
lbracks_re = re.compile(r'[\(\[\{\<]')
rbracks_re = re.compile(r'[\)\]\}\>]')
special_re = re.compile(r'\\')

def normalize(text):
    text = spaces_re.sub(' ', text)
    text = lbracks_re.sub('(', text)
    text = rbracks_re.sub(')', text)
    text = special_re.sub('', text)
    #text = ''.join(filter(lambda x: ord(x) > 127, list(text)))
    text = unidecode(text)
    text = text.lower()
    return text
	
	train_df = pd.read_csv('../input/train.csv')[['question_text', 'target']]
train_df['question_text'] = train_df['question_text'].apply(normalize)

test_df = pd.read_csv('../input/test.csv')[['question_text', 'qid']]
test_df['question_text'] = test_df['question_text'].apply(normalize)
test_df['index'] = test_df.index

train = train_df.values
test = test_df[['question_text', 'index']].values

frac_marked = len(train_df[train_df['target'] == 1]) / len(train_df)

class_weights = {
    0: 1 / (1 - frac_marked),
    1: 1 / frac_marked
}

plt.hist(list(map(len, train_df['question_text'])), bins=50, range=(0, 400))
plt.show()


batch_size = 128

text_field = Field(tokenize=list, fix_length=fix_length)
label_field = Field(sequential=False, use_vocab=False)

fields = [
    ('text', text_field),
    ('label', label_field)
]

examples_train = [Example.fromlist(datum, fields) for datum in train]
examples_test = [Example.fromlist(datum, fields) for datum in test]

train_dataset = Dataset(examples_train, fields)
test_dataset = Dataset(examples_test, fields)

train_dataset, val_dataset = train_dataset.split(0.95)

text_field.build_vocab(train_dataset)

sort_key = lambda x: len(x.text)

train_it, val_it = BucketIterator.splits([train_dataset, val_dataset], batch_size=batch_size, repeat=True, sort_key=sort_key, shuffle=True)
test_it = BucketIterator(test_dataset, batch_size=batch_size, sort_key=sort_key)

train_gen, val_gen, test_gen = WrapIterator.wraps([train_it, val_it, test_it], permute={'text': (1, 0)}, x_fields=['text'], y_fields=['label'])

vocab_size = len(text_field.vocab)

model.add(InputLayer(input_shape=(None,)))
model.add(Lambda(lambda x: K.one_hot(K.cast(x, dtype='int32'), num_classes=vocab_size)))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

batches = list(tqdm(islice(train_gen, len(train_gen)), total=len(train_gen)))

model.fit_generator(iter(batches), steps_per_epoch=len(train_gen), epochs=3, class_weight=class_weights)

def evalute(data_gen):
    truth, pred = [], []
    for batch_x, batch_y in tqdm(islice(data_gen, len(data_gen)), total=len(data_gen)):
        pred += model.predict_on_batch(batch_x)[:, 0].round().tolist()
        truth += batch_y[0].tolist()
    return balanced_accuracy_score(truth, pred)
	
train_acc = evalute(train_gen)
print('Train acc: {}'.format(train_acc))

val_acc = evalute(val_gen)
print('Val acc: {}'.format(val_acc))

qids, pred = [], []
for x, y in tqdm(test_gen):
    pred += model.predict_on_batch(x)[:, 0].round().astype('int').tolist()
    qids += list(map(lambda i: test_df.iloc[i]['qid'], y[0]))
	
sub = pd.DataFrame({'qid': qids, 'prediction': pred})
sub.to_csv('submission.csv', index=False)
sub.to_csv('submission.csv', index=False)