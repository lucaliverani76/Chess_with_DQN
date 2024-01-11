#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import random
import numpy as np
import collections
import pickle
import os


import glob
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



import Chessx
import matplotlib.pyplot as plt



# sole purpose of this function is to upload data from previous Colab session
def findoldcalcs(extn):
    txtfiles = []

    for file in glob.glob(extn):
        txtfiles.append(file)

    x=[]
    for txt in txtfiles:
        x=x+re.findall(r'\d+', txt)
    #print(x)
    x=[int(y) for y in x]
    if len(x)>0:
        result=[max(x)]
    else:
        result=[]

    if len(result)>0:
        for n,y in enumerate(x):
            #print(n,y)
            if y!=max(x):
              #print(txtfiles[n])
              os.remove(txtfiles[n])
        return int(result[0])
    else:
        return int(0)

#this function looks if previous sessions of colab crushed and restart the system from there
def Dogenericstuff(Q_action_value):
    Savedweights_=findoldcalcs(Q_action_value.name+"*.pth")
    episode=np.floor(Savedweights_/1000)
    minibatch_sample=Savedweights_-episode*1000

    if episode!=0 or minibatch_sample!=0:
        # loading weights
        Q_action_value.wpath=Q_action_value.name+str(int(episode*1000+minibatch_sample))+'.pth'
        #print(Q_target_value.wpath)
        Q_action_value.loadWeights()
        Chessx.copyweightsfromto_Q_action_value_to_Q_target_value()

    return episode, minibatch_sample


def Savingweights():
    # saving weights
    Chessx.Q_action_value.wpath=Chessx.Q_action_value.name+str(int(episode*1000+minibatch_sample))+'.pth'
    #print(Q_action_value.wpath)
    Chessx.Q_action_value.SaveWeights()


def Defineorload_reply_memory(N):
    n=findoldcalcs("*.pickle")  # sole purpose of taking over where we left
    # print(n)

    if n>0:
      with open('Replaymemory'+str(n)+'.pickle', 'rb') as handle:
          D = pickle.load(handle)
    else:
        D=collections.deque(maxlen=N)
    return n,D

def Save_reply_memory():
    if (n)%1000==0:
        print("Replay memory reached {} units".format(n))
        with open('Replaymemory'+str(n)+'.pickle', 'wb') as handle:
            pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

def ResetReplymemory():
    txtfiles = []
    for file in glob.glob("*.pickle"):
        txtfiles.append(file)
    for f in txtfiles:
        os.remove(f)

def ResetWeight_History():
    txtfiles = []
    for file in glob.glob("*.pth"):
        txtfiles.append(file)
    for f in txtfiles:
        os.remove(f)

def Trytogetoldweights():

    try:
      Chessx.Q_action_value.wpath=Chessx.Q_action_value.name +'.pthx'
      Chessx.Q_action_value.loadWeights()
      Chessx.copyweightsfromto_Q_action_value_to_Q_target_value()
      print("loaded old dictionary")

    except:
      pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# =============================================================================
# Algorithm DQN
# =============================================================================

T=50 #number of action steps, or games moves


epsilon=0.9 #greedy policy
decay = 0.95
min_epsilon = 0.2



 # initialize replay memory with size N
N=1000

lambda_=0.3 #discount factor

N_minibatch_sampling=1000

n_tests_on_batch=int(N/N_minibatch_sampling*1.5)

N_episodes=2

# weightfile='model_weights_tanh_moreneurons_Hdiscountf'

torch.set_printoptions(precision=10)




Chessx.Q_action_value, Chessx.Q_target_value=Chessx.Define_Model_to_train(1)




Chessx.initializeAgents()  #simply we initialize the weights of value functions


Trytogetoldweights()





episode, minibatch_sample= Dogenericstuff(Chessx.Q_action_value)




while episode < N_episodes:

    print("episode {0}".format(episode))

    # I fill the Replay memory
    n,D=Defineorload_reply_memory(N)


    while n<N:

        #sequences can start from always the initial
        #state or in a random state, depending how we build the fuction initialize_sequence()

        S=Chessx.initialize_sequences()
        white_has_to_move=True



        for j in range(T):

            possible_actions=Chessx.PossibleActions(S)
            
            if (possible_actions==[] or possible_actions==None):
                break

            if random.uniform(0, 1)<epsilon:
                #especially at the beginning where we have no experince we tend to choose random actions
                action=possible_actions[random.randrange(len(possible_actions))]
            else:
                #we evaluate the best possible action given the current state
                action_values =Chessx.eval_Q_action_value(S,possible_actions)

                if white_has_to_move:
                    i=np.argmax(action_values)
                else:
                    i=np.argmin(action_values)

                action=possible_actions[i]

            S_next=Chessx.Implement_action(S,action)

            reward=Chessx.Reward(S)

            if not white_has_to_move:  #white has just moved we record the reward of his move hoping that it was a good choice
                s_=Chessx.initialize_sequences()
                Chessx.Copy_board_state(s_,S)
                s_n=Chessx.initialize_sequences()
                Chessx.Copy_board_state(s_n,S_next)
                D.append([s_,action,reward,s_n])
                n=n+1
                Save_reply_memory()



            S=S_next



            white_has_to_move=not white_has_to_move

            if S.loc[0,"isover"]:
                break;



    print("End replay memory")
        # current_replay_memory_ratio=int(np.ceil(len(D)/N_minibatch_sampling))





    while minibatch_sample < n_tests_on_batch:

        print("Test io  minibatch {0}".format(minibatch_sample))

        S_s,action_s,reward_s,S_next_s =Chessx.extract_minibatch(D, N_minibatch_sampling)  #extract data from batch

        y = (reward_s.reshape(-1,1)+ lambda_* Chessx.Max_Q_target_value(S_next_s).reshape(-1,1)).astype(np.float32)
        

        # trying to optimize the fit between Q_action_value and y
        R=Chessx.OptimizeQ_action_value(S_s,y)


        Savingweights()

        minibatch_sample=minibatch_sample+1


    minibatch_sample=0

    #at the end of every training we align the target and action value function
    # hoping that these asyntotic optimization will converge on something meaningful
    Chessx.copyweightsfromto_Q_action_value_to_Q_target_value()

    # I start reducing the random moves and I follow more
    #and more the Q function advises
    epsilon = max(min_epsilon, epsilon*decay)

    n=0
    ResetReplymemory()


    episode=episode+1





#And finally
ResetWeight_History()
Chessx.Q_action_value.wpath=Chessx.Q_action_value.name+'.pthx'
Chessx.Q_action_value.SaveWeights()


# action_list=[]
# for Snx in S_s:
#     v=np.hstack((Snx.board_np[0][0].copy().reshape(1,-1).squeeze(),np.array([Snx.im[0],Snx.ckm[0],Snx.ck[0], Snx.score[0]])))
#     v=v.astype(np.float32)
#     action_list.append(v)
# AL=np.array(action_list)
# # print(AL)
# action_values_=(Chessx.Q_action_value.forward(torch.tensor(AL, device=params['device'])).cpu().numpy())









# Plot input and output
output_np = R.detach().numpy()

R_=reward_s.reshape(-1,1)
plt.figure(figsize=(10, 5))


plt.plot(output_np, label='NN')

plt.plot(y, label='Q function')
plt.plot(R_, label='Rewards')

# plt.plot(action_values_, label='Bo')
plt.title('Input Data')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')

agent=Chessx.Q_action_value
sample_layer_weights = agent.state_dict()

sample_layer_weights2 = Chessx.Q_action_value.state_dict()


# =============================================================================
# ________________________________________________________
# =============================================================================

