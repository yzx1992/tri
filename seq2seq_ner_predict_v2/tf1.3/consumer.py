#coding: utf8
from datautil import HotIter
import pika
DEBUG = False
name = 'hello'
if DEBUG:
    conn = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = conn.channel()
    channel.queue_declare(queue=name)
    # def callback(ch, method, properties, body):
    #     print('[x] Received {}'.format(body))
    # channel.basic_consume(callback,queue=name,no_ack=True)
    # channel.start_consuming()
    for method_frame, properties, body in channel.consume(name):
        print('[x] Received {}'.format(body))
else:
    iter = HotIter('data/train.crf.ids.txt', mq_host='localhost', queue_name=name)

    for w,x,fea,term,y in iter:
        print('--------')
        print(w)
        print(x)
        print(fea)
        print(term)
        print(y)
