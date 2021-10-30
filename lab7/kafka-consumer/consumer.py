from kafka import KafkaConsumer, TopicPartition

size = 1000000


def read_from_topic(kafka_consumer, topic):
    kafka_consumer.subscribe([topic])
    for msg in kafka_consumer:
        print(msg)


def read_from_topic_with_partition(kafka_consumer, topic1, topic2):
    kafka_consumer.assign([TopicPartition(topic1, 1), TopicPartition(topic2, 1)])
    for msg in kafka_consumer:
        print(msg)


def read_from_topic_with_partition_offset(kafka_consumer, topic):
    partition = TopicPartition(topic, 0)
    kafka_consumer.assign([partition])
    last_offset = kafka_consumer.end_offsets([partition])[partition]
    for msg in kafka_consumer:
        if msg.offset == last_offset - 1:
            break


if __name__ == '__main__':
    consumer = KafkaConsumer(bootstrap_servers='34.68.84.17:9092')
    read_from_topic(consumer, 'topic')
