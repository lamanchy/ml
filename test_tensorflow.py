# coding=utf-8


def test_tensorflow():
    import tensorflow as tf

    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


if __name__ == "__main__":
    test_tensorflow()
