from kafka import KafkaProducer


def kafka_python_producer_sync(producer, size, msg):
    for _ in range(size):
        future = producer.send('topic', msg)
        result = future.get(timeout=60)
    producer.flush()


def success(metadata):
    print(metadata.topic)


def error(exception):
    print(exception)


def kafka_python_producer_async(producer, size, msg):
    for _ in range(size):
        producer.send('topic', msg).add_callback(success).add_errback(error)
    producer.flush()


if __name__ == '__main__':
    producer = KafkaProducer(bootstrap_servers='34.68.84.17:9092')
    kafka_python_producer_sync(producer, 100, "Hello World")
