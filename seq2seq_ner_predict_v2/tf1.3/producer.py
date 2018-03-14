#coding: utf8
from __future__ import print_function
import pika
import time
from datautil import Itertool

DEBUG = False
name = 'hello'
if DEBUG:
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=name)

    channel.basic_publish(exchange='', routing_key=name, body='Hello World!')
    print(" [x] Sent 'Hello World!'")
    connection.close()
else:
    conn = pika.BlockingConnection()
    channel = conn.channel()
    # channel.queue_declare(queue=name, durable=True)
    channel.queue_declare(queue=name)
    with open('data/valid.crf.ids.txt', 'r') as f:
        sample_num = 0
        samples = []
        # words = []
        for line in f:
            line = line.rstrip('\n')
            sample_num += 1
            word = line.split('\t')
            assert len(word) == 4
            # channel.basic_publish(exchange='', routing_key=word[0], body=line, properties=pika.BasicProperties(delivery_mode=2))
            channel.basic_publish(exchange='', routing_key=name, body=line)
            print('cnt:{}'.format(sample_num))
            time.sleep(0.3)
            if sample_num % 10 == 0:
                channel.basic_publish(exchange='', routing_key=name, body='_END')

