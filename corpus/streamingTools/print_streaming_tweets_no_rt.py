import json
import argparse


def print_streaming_tweets(streaming_path):

    # Abro el txt de tweets capturados por streaming.py
    with open(streaming_path, 'r') as file:
        for cnt, line in enumerate(file):
            tweet = json.loads(line)
            if 'user' not in tweet.keys():
                continue
            # Quito ReTweets y metadata.
            text = tweet['text']
            if text.find("RT ") == -1:
                # Printeo y desde consola los mando a un .txt nuevo
                print(text)


parser = argparse.ArgumentParser(description='Streaming Tweets')
parser.add_argument('-f', '--file', dest='file_path',
                    help='Text raw file path file')
args = parser.parse_args()

print_streaming_tweets(args.file_path)
