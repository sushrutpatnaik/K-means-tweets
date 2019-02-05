
#Compute the similarity between tweets using the Jaccard Distance metric.
#Cluster tweets using the K-means clustering algorithm

import pandas as pd
import sys
import json

def getKeyWords(sentence):
    wordList = sentence.split(" ")
    return list(set(wordList))


def storeTweetText(text):
    text_words = getKeyWords(text)
    word_dict = dict()
    for word in text_words :
        word_dict[word] = True
    return word_dict

def findCentroids(path):
    with open(path, 'r') as myfile:
        tweetData = myfile.read().replace('\n', '')
    return tweetData.split(",")




def findJaccard(text1, text2):
    tweetText1 = storeTweetText(text1)
    tweetText2 = storeTweetText(text2)
    union_dict = tweetText1.copy()
    union_dict.update(tweetText2)
    union = len(union_dict)
    tweetKeys1 = set(tweetText1.keys())
    tweetKeys2 = set(tweetText2.keys())
    intersection = len(tweetKeys1 & tweetKeys2)
    distance_text =1 - (intersection / union)
    return (distance_text)

def updateCentroid(cluster, tweet_data):
    updated_cluster = dict()
    for center in cluster.keys():
        min_distance = 999
        min_center = center
        for temp_center in cluster[center]:
            total_distance = 0
            for id in cluster[center]:
                if temp_center != id:
                    total_distance += findJaccard(tweet_data[temp_center], tweet_data[id])
            if min_distance > total_distance:
                min_distance = total_distance
                min_center = temp_center
        updated_cluster[min_center] = cluster[center]
    return updated_cluster

def SSE(cluster, tweet_data):
    sum = 0
    for center in cluster.keys():
        for id in cluster[center]:
            if center != id:
                sum += findJaccard(tweet_data[center], tweet_data[id])
    return sum

noOfClusters = sys.argv[1]
initial_seeds = sys.argv[2]
tweets_input_path = sys.argv[3]
output_path = sys.argv[4]
df = pd.read_json(tweets_input_path,lines=True)
centroid = findCentroids(initial_seeds)
id_list = df['id']
tweet_list = df['text']
tweet_data = dict()
for tweetId, tweet in zip(id_list, tweet_list):
    tweet_data[str(tweetId)] = tweet
temp = True
while(temp):
    cluster = dict()
    for tweetId in tweet_data.keys():
        if tweetId not in centroid:
            minDistance = 9999
            minCenterID = ""
            for centerID in centroid:
                jaccardDistance = findJaccard(tweet_data[tweetId], tweet_data[centerID])
                if minDistance > jaccardDistance:
                    minDistance = jaccardDistance
                    minCenterID = centerID
            if not cluster.get(minCenterID):
                cluster[minCenterID] = []
                cluster[minCenterID].append(minCenterID)
            cluster[minCenterID].append(tweetId)
    modifiedCluster = updateCentroid(cluster, tweet_data)
    oldCluster_keys = set(cluster.keys())
    modifiedCluster_keys = set(modifiedCluster.keys())
    intersection_dict = len(oldCluster_keys & modifiedCluster_keys)
    if intersection_dict == len(cluster.keys()):
        temp = False
    else:
        centroid = modifiedCluster.keys()
noOfIterations = 1
for key in cluster.keys():
    with open(output_path, 'a') as f1:
        f1.write(str(noOfIterations) + ' ' + ",".join(cluster[key]) + "\n")
    noOfIterations+=1
with open(output_path, 'a') as f1:
    f1.write('Sum of Squared Errors ' + str(SSE(cluster, tweet_data)))


