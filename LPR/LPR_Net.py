import tensorflow as tf
import numpy as np

#Test parameters
BATCH_SIZE = 1    #Batch size for train
test_numbers = 100

#Data parameters
testi='parking_t/'   #Test data path
img_size = [94, 24]  #Image size
num_channels = 1
label_len = 7

#Possible characters in Korean plate
CHARS = ['가', '나', '다', '라', '마','아', '바', '사', '자', '하',
         '거', '너', '더', '러', '머', '버', '서', '어', '저', '허',
         '고', '노', '도', '로', '모', '보', '소', '오', '조', '호',
         '구', '누', '두', '루', '무', '부', '수', '우', '주',
         '배',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'
        ]

#Character dictionary
dict = {'A01':'가', 'A02':'나', 'A03':'다', 'A04':'라', 'A05':'마',
        'A06':'바', 'A07':'사', 'A08':'아', 'A09':'자', 'A10':'하',
        'B01':'거', 'B02':'너', 'B03':'더', 'B04':'러', 'B05':'머',
        'B06':'버', 'B07':'서', 'B08':'어', 'B09':'저', 'B10':'허',
        'C01':'고', 'C02':'노', 'C03':'도', 'C04':'로', 'C05':'모',
        'C06':'보', 'C07':'소', 'C08':'오', 'C09':'조', 'C10':'호',
        'D01':'구', 'D02':'누', 'D03':'두', 'D04':'루', 'D05':'무',
        'D06':'부', 'D07':'수', 'D08':'우', 'D09':'주', 'D10':'배',
       }

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
NUM_CHARS = len(CHARS)

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = CHARS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def small_basic_block(x,im,om):
    x = conv(x,im,int(om/4),ksize=[1,1])
    x = conv(x,int(om/4),int(om/4),ksize=[3,1],pad='SAME')
    x = conv(x,int(om/4),int(om/4),ksize=[1,3],pad='SAME')
    x = conv(x,int(om/4),om,ksize=[1,1])
    return x

def conv(x,im,om,ksize,stride=[1,1,1,1],pad = 'SAME'):
    conv_weights = tf.Variable(
        tf.truncated_normal([ksize[0], ksize[1], im, om],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=None, dtype=tf.float32))
    conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32))
    out = tf.nn.conv2d(x,
                        conv_weights,
                        strides=stride,
                        padding=pad)
    relu = tf.nn.bias_add(out, conv_biases)
    return relu


# Paper link https://arxiv.org/pdf/1806.10447.pdf
def get_train_model(num_channels, label_len, b, img_size):
    inputs = tf.placeholder(
        tf.float32,
        shape=(b, img_size[0], img_size[1], num_channels))

    # targets = tf.sparse_placeholder(tf.int32)

    seq_len = tf.placeholder(tf.int32, [None])

    x = inputs

    x = conv(x, num_channels, 64, ksize=[3, 3])
    x = tf.layers.batch_normalization(x, training=True)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 1, 1, 1],
                       padding='SAME')
    x = small_basic_block(x, 64, 64)
    x2 = x
    x = tf.layers.batch_normalization(x, training=True)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')

    x = small_basic_block(x, 64, 256)
    x = tf.layers.batch_normalization(x, training=True)
    x = tf.nn.relu(x)

    x = small_basic_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x, training=True)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')

    x = tf.layers.dropout(x)

    x = conv(x, 256, 256, ksize=[4, 1])
    x = tf.layers.dropout(x)

    x = tf.layers.batch_normalization(x, training=True)
    x = tf.nn.relu(x)

    x = conv(x, 256, NUM_CHARS + 1, ksize=[1, 13], pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x, cx)

    x1 = tf.nn.avg_pool(inputs,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)

    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)

    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

    x = tf.concat([x, x1, x2, x3], 3)
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))
    logits = tf.reduce_mean(x, axis=2)

    return logits, inputs, seq_len
    # return logits, inputs, targets, seq_len