#!/usr/bin/python3

import sys
import random
from math import log
import numpy as np
from operator import itemgetter


#Iused this resource to understand the logic: https://en.wikipedia.org/wiki/Viterbi_algorithm

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 30
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
# <Your feedback goes here>
#I had to look up online to understand viterbi algorithm properly
#I was not able to attend last few lectures due to my health and there areno slides on viterbi algorithm online
#####################################################
#####################################################



# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0

class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior = [0.5, 0.5]
        self.transition = [[0.999, 0.001], [0.01, 0.99]]
        self.emission = [{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                         {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}]

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Computes the (natural) log probability of sequence given a sequence of states.
    def logprob(self, sequence, states):
        #We will enumerate the set of states and calculate the probability of sequence
        #For each state, transition and emission probability will be computed
        #To take the log of probability, we will follow formula: log(ab) = log(a) + log(b)
        prob =0
        for index, val in enumerate(states):
            if index == 0:
                prob = log(self.emission[val][sequence[index]]) + log(self.prior[index])
            elif index == 1:
                probability = log(self.transition[states[0]][val]) + log(self.emission[val][sequence[index]])
                prob = prob+probability
            elif index>1:
                probability = log(self.transition[states[index-1]][val]) + log(self.emission[val][sequence[index]])
                prob = prob+probability
        return prob
    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]


    def viterbi(self, sequence):

        #make a list of dict as a table for viterbi algorithm
        viterbiTable=[{}]
        lenSequence = len(sequence)
        states=[0,1] #since we know 2 states are given, so it is 0 and 1

        for i in states:
            #print("index", idx, "value", val)
            #calculating the sum of probabilties in every state
            viterbiTable[0][i]={"probability":log(self.prior[i])+log(self.emission[i][sequence[0]]),"prevState":0}


        for index in range(1,lenSequence):

            viterbiTable.append({})
            for st in states:
                #will have probailty and prevState keys in the dictionary
                #to keep track of the present proabilty and backpointers that we used to save the state
                probMax=max(viterbiTable[index-1][s1]["probability"]+log(self.transition[s1][st]) for s1 in states)
                for s1 in states:
                    if viterbiTable[index-1][s1]["probability"]+log(self.transition[s1][st])==probMax:
                        probMax=probMax+log(self.emission[st][sequence[index]])
                        viterbiTable[index][st]={"probability":probMax,"prevState":s1}
                        break

        #listStateIndices=[]
        listStateIndices=[]

        #for item in viterbiTable[-1].values():
        ListofAllProb = list(value["probability"] for value in viterbiTable[-1].values())
        max_prob = max(ListofAllProb)

        #max_prob = max(map(itemgetter("probability"), viterbiTable))
        #max_prob = max(prob)
        lstate = 0
        for state, prob in viterbiTable[-1].items():
            if prob["probability"] == max_prob:
                listStateIndices.append(state)
                break

            lstate = state

        tableLength = len(viterbiTable)
        for t in range(tableLength-2, -1, -1):
            Tablevalue = viterbiTable[t + 1][lstate]["prevState"]
            listStateIndices.append(Tablevalue)
            lstate = viterbiTable[t + 1][lstate]["prevState"]

        counter0= listStateIndices.count(0)
        counter1 = listStateIndices.count(1)

        #print ("line1",self.logprob(sequence, listStateIndices))
        #print ("zeroesss: ", counter0)
        #print ("Onesssss: ", counter1)
        #print(listStateIndices)
        return listStateIndices


def read_sequence(filename):
    with open(filename, "r") as f:
        return f.read().strip()

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, logprob, states):
    with open(filename, "w") as f:
        f.write(str(logprob))
        f.write("\n")
        for state in range(2):
            f.write(str(states.count(state)))
            f.write("\n")
        f.write("".join(map(str, states)))
        f.write("\n")

hmm = HMM()

file = sys.argv[1]
sequence = read_sequence(file)
viterbi = hmm.viterbi(sequence)
logprob = hmm.logprob(sequence, viterbi)
name = "my_"+file[:-4]+'_output.txt'
write_output(name, logprob, viterbi)
