#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Created on Sat Oct 14 19:26:02 2023

@author: luca
"""

import chess
import chess.svg



import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pandas as pd



# Here I simply gives some additional scores to some situations to add to the value function
CHECKMATE=20
SIMPLECHECK=3
ADVANTAGE_INSUFFICIENT_MATERIAL=6

TOTAL_COLOR_SCORE=37 * 2 #this is just the sum of the value assigned to every piece by the chess library
# array([[ 4.,  2.,  3.,  5.,  6.,  3.,  2.,  4.],
#        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [-1., -1., -1., -1., -1., -1., -1., -1.],
#        [-4., -2., -3., -5., -6., -3., -2., -4.]], dtype=float32)



DATAINPUT_SIZE=8*8 + 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def PossibleActions(board_state):
    board = chess.Board()
    board.set_fen(board_state.board[0])
    move1=board.legal_moves
    listofmoves=[str(move) for move in move1]
    return listofmoves

def drawboard(board_state):
    board = chess.Board()
    board.set_fen(board_state.board[0]) 
    print(board) 
    
# =============================================================================
# Thanks niklas
# https://github.com/niklasf/python-chess/issues/404
#
# =============================================================================
def convert_to_num(board):
        l = [None] * 64
        for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):  # Check if white
            l[sq] = board.piece_type_at(sq)
        for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):  # Check if black
            l[sq] = -board.piece_type_at(sq)
        new_list = [0 if x is None else x for x in l]
        temp=  [np.array(new_list).reshape(-1,8).astype(np.float32)]
        return temp



def initialize_sequences(*args):

    if len(args) == 1 and isinstance(args[0], np.ndarray):
        board_state=pd.DataFrame(columns=["board","board_np","im","ckm","ck","isover", "score"])
        board_state.loc[0] = [args[0][0],args[0][1],args[0][2],args[0][3],args[0][4],args[0][5], args[0][6]]

    if len(args)==0:
        board = chess.Board()
        board_state=pd.DataFrame(columns=["board","board_np","im","ckm","ck","isover", "score"])
        board_state.loc[0] = [board.fen(),convert_to_num(board),
                              float(board.is_insufficient_material()),float(board.is_checkmate()),float(board.is_check())\
                                  ,float(board.is_insufficient_material()  or board.is_checkmate()),0 ]
        board_state.loc[0, "score"]= Reward(board_state)
    return board_state


def fromChesstoS(board):
        
    board_state=pd.DataFrame(columns=["board","board_np","im","ckm","ck","isover","score"])
    board_state.loc[0] = [board.fen(),convert_to_num(board),
                          float(board.is_insufficient_material()),float(board.is_checkmate()),float(board.is_check())\
                              ,float(board.is_insufficient_material()  or board.is_checkmate()), 0]
    board_state.loc[0, "score"]= Reward(board_state)
    
    return board_state


def Implement_action(board_state,Action):
    B_result=initialize_sequences()
    board = chess.Board()
    board_state_fen=board_state.board[0]
    board.set_fen(board_state_fen)
    board.push_san(Action)
    B_result.loc[0,"board_np"]=convert_to_num(board)
    B_result.loc[0,"im"]=float(board.is_insufficient_material())
    B_result.loc[0,"ckm"]=float(board.is_checkmate())
    B_result.loc[0,"ck"]=float(board.is_check())
    B_result.loc[0,"isover"]=float(board.is_insufficient_material()  or board.is_checkmate())
    B_result.loc[0,"board"]=board.fen()
    B_result.loc[0, "score"]= Reward(B_result)
    return B_result




def Copy_board_state(board_state,board_state_source):
    L=board_state_source.values.tolist()[0].copy()
    # print((board_state.loc[0,"board_np"]).shape)
    board_state.loc[0,"board"]=L[0]
    # print((L[1].copy()).shape)
    board_state.loc[0,"board_np"]=L[1]
    board_state.loc[0,"im"]=L[2]
    board_state.loc[0,"ckm"]=L[3]
    board_state.loc[0,"ck"]=L[4]
    board_state.loc[0,"isover"]=L[5]
    board_state.loc[0, "score"]= L[6]



def Reward(board_state):

    board_np=board_state.loc[0,"board_np"]

    board_np=np.array(board_np).astype(int)
    
    reward=np.sum(board_np).astype(np.float32)/np.sum(np.abs(board_np.astype(np.float32))) +\
    board_state.loc[0,"ck"]*SIMPLECHECK/np.sum(np.abs(board_np.astype(np.float32))) \
        + board_state.loc[0,"im"]*ADVANTAGE_INSUFFICIENT_MATERIAL/np.sum(np.abs(board_np.astype(np.float32)))\
            + board_state.loc[0,"ckm"]*ADVANTAGE_INSUFFICIENT_MATERIAL/np.sum(np.abs(board_np.astype(np.float32)))

    return reward/100




def flip_ranks(board):
    # Convert the board to FEN string
    fen = board.fen()

    # Split the FEN string into its components
    pieces, turn, castling, en_passant, halfmove, fullmove = fen.split(' ')

    # Reverse the order of the ranks and files
    pieces = '/'.join([''.join(reversed(rank)) for rank in pieces.split('/')])

    # Construct the new FEN string
    new_fen = f"{pieces} {turn} {castling} {en_passant} {halfmove} {fullmove}"

    # Create a new board with the flipped ranks and files
    new_board = chess.Board(fen=new_fen)

    return new_board


def define_parameters():
    params = dict()

    params['learning_rate'] = 0.0001

    params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    return (params)




class DQNAgent(torch.nn.Module):
    
    def __init__(self, params):

        super().__init__()

        self.learning_rate = params['learning_rate']
        self.inputsize=DATAINPUT_SIZE-1
        self.first_layer = 200
        self.second_layer = 100
        self.third_layer = 50
        self.fourth_layer = 1
        self.fifth_layer = 1


        self.name="DQNAgent"
        self.wpath=self.name+".pthx"

        self.optimizer = None
        self.network()

    def network(self):
        # Layers
        self.f1 = nn.Linear(self.inputsize, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 1)

    def loadWeights(self):
        self.load_state_dict(torch.load(self.wpath,map_location=torch.device(params['device'])))


    def SaveWeights(self):
        torch.save(self.state_dict(), self.wpath)



    def forward(self, x):
        
        x_first = x[:,:-1]


        # Extract the last element
        x_last = x[:,-1].view(-1,1)

        #y = torch.tanh(self.f1(x_first))
        y = F.elu(self.f1(x_first))
        y= F.elu(self.f2(y))
        y = F.elu(self.f3(y)) #*x_last
        y = self.f4(y)
        return y





class DQNAgent_morebrain(DQNAgent):
    
    def __init__(self, params):

        super().__init__(params)
        
        self.name="DQNAgent_morebrain"
        self.wpath=self.name+".pthx"  
        self.first_layer = 1000
        self.second_layer = 1600
        self.third_layer = 500
        self.fourth_layer = 100
        self.fifth_layer = 20
        self.inputsize=DATAINPUT_SIZE-1
        self.network()
        
    def network(self):
        super().network()
        # Layers
        self.f1 = nn.Linear(self.inputsize, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, self.fourth_layer)
        self.f5 = nn.Linear(self.fourth_layer, self.fifth_layer)
        self.f6 = nn.Linear(self.fifth_layer, 1)
        
        
    def forward(self, x):
        
        x_first = x[:,:-1]


        # Extract the last element
        x_last = x[:,-1].view(-1,1)

        #y = torch.tanh(self.f1(x_first))
        y = F.elu(self.f1(x_first))
        y= F.elu(self.f2(y))
        y = F.elu(self.f3(y)) #*x_last
        y = F.elu(self.f4(y)) #*x_last
        y = F.elu(self.f5(y)) #*x_last
        y = self.f6(y)
        return y
        



params=define_parameters()

def Define_Model_to_train(n):
    if n==0:

        Q_action_value=DQNAgent(params).to(params['device'])
        Q_target_value=DQNAgent(params).to(params['device'])
        
            
    if n==1:

        Q_action_value=DQNAgent_morebrain(params).to(params['device'])
        Q_target_value=DQNAgent_morebrain(params).to(params['device'])
        
    return Q_action_value, Q_target_value


Q_action_value, Q_target_value=Define_Model_to_train(1)


def initializeAgents():
    for layer_ in Q_action_value.children():
        # torch.nn.init.uniform_(layer_.weight,-10.0, 10.0).to(params["device"])
        torch.nn.init.xavier_normal_(layer_.weight).to(params["device"])
        # nn.init.zeros_(layer_.bias).to(params["device"])
        nn.init.constant_(layer_.bias, 0).to(params["device"])


    Q_target_value.load_state_dict(Q_action_value.state_dict())
    for param in Q_target_value.parameters():
        param.requires_grad = False


def copyweightsfromto_Q_action_value_to_Q_target_value():
    Q_target_value.load_state_dict(Q_action_value.state_dict())



def eval_Q_action_value(S,possible_actions):
    if (possible_actions==[] or possible_actions==None):
        return [0]
    torch.set_grad_enabled(False)
    action_list=[]
    for action in possible_actions:
        Snx=Implement_action(S,action)
        v=np.hstack((Snx.board_np[0][0].copy().reshape(1,-1).squeeze(),np.array([Snx.im[0],Snx.ckm[0],Snx.ck[0], Snx.score[0]])))
        v=v.astype(np.float32)
        action_list.append(v)
    AL=np.array(action_list)
    action_values=(Q_action_value.forward(torch.tensor(AL, device=params['device'])).cpu().numpy())
    return action_values


def eval_Q_target_value(S,possible_actions):
    if (possible_actions==[] or possible_actions==None):
        return [0]
    torch.set_grad_enabled(False)
    action_list=[]
    for action in possible_actions:
        Snx=Implement_action(S,action)
        v=np.hstack((Snx.board_np[0][0].copy().reshape(1,-1).squeeze(),np.array([Snx.im[0],Snx.ckm[0],Snx.ck[0],Snx.score[0]])))
        v=v.astype(np.float32)
        action_list.append(v)
    AL=np.array(action_list)

    action_values=(Q_target_value.forward(torch.tensor(AL, device=params['device'])).cpu().numpy())
    return action_values




def extract_minibatch(D, length_batch):
    array_=np.array(random.sample(D, length_batch),     dtype=object)
    # columns_tuple = tuple(map(np.array,zip(*array_)))

    return array_[:,0], array_[:,1], array_[:,2], array_[:,3]


def Max_Q_target_value(Ss):

    Max_Q_t_values=[]

    for S in Ss:
        
        board = chess.Board()
        board.set_fen(S.loc[0,"board"])

        mirrored_board = flip_ranks(board.mirror())
        S_black=fromChesstoS(mirrored_board)
        possible_actions=PossibleActions(S_black)
            

        if (possible_actions==[] or possible_actions==None) or (np.abs(S_black.loc[0,"score"])>0.0015):
            Max_Q_t_values.append(0)
            continue
        target_values =eval_Q_target_value(S_black,possible_actions)
        # print(target_values)
        i=np.argmax(target_values)
        
        mirrored_board.push_san(possible_actions[i])    
        re_mirrored_board = flip_ranks(mirrored_board.mirror())
        B_next=fromChesstoS(re_mirrored_board)  #actually we know that the black will play against us
        
        
        possible_actions=PossibleActions(B_next)
        target_values =eval_Q_target_value(B_next,possible_actions)
        # i=np.argmax(target_values)   #once the black has played the worst move for us, we have to fight back
        
        Max_Q_t_values.append(np.amax(target_values))
        
        
    return np.array(Max_Q_t_values)





def OptimizeQ_action_value(S_s,y):
    agent=Q_action_value
    agent.optimizer=optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    target_f=torch.tensor(y, device=params['device']).float()


    R=[]
    for Snx in S_s:
        v=np.hstack((Snx.board_np[0][0].copy().reshape(1,-1).squeeze(),np.array([Snx.im[0],Snx.ckm[0],Snx.ck[0], Snx.score[0]])))
        v=v.astype(np.float32)
        R.append(v)
    x_data=torch.tensor(np.array(R),device=params['device']).float()
    # print(x_data)
    # print(x_data.shape)
    agent.train()
    torch.set_grad_enabled(True)
    num_epochs = 10
    for epoch in range(num_epochs):
        output=agent.forward(x_data)


        # print(output)
        agent.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        agent.optimizer.step()

        # Print the loss every 100 epochs
    print(f'Loss: {loss.item()}')
    return output




def eval_Q_value_generic(Q,S,possible_actions):
    if (possible_actions==[] or possible_actions==None):
        return [0]
    torch.set_grad_enabled(False)
    action_list=[]
    for action in possible_actions:
        Snx=Implement_action(S,action)
        v=np.hstack((Snx.board_np[0][0].copy().reshape(1,-1).squeeze(),np.array([Snx.im[0],Snx.ckm[0],Snx.ck[0], Snx.score[0]])))
        v=v.astype(np.float32)
        action_list.append(v)
    AL=np.array(action_list)
    # print(AL)
    action_values=(Q.forward(torch.tensor(AL, device=params['device'])).cpu().numpy())

    return action_values




# =============================================================================
# 
# °°°°°°°°°°°°°°°°°°°°°°
# 
# SIMPLE MECHANICAL TURK
# 
# °°°°°°°°°°°°°°°°°°°°°°
# =============================================================================


def basefunction_mt(S,n,R,L,choice):
    possible_actions=PossibleActions(S)


    if (possible_actions==[] or possible_actions==None) or n==0:
        R1=R
        L.append([R,choice])    
        return 0
  
    for ch,a in enumerate(possible_actions):        
        S_next=Implement_action(S,a)
        if n==1:

            basefunction_mt(S_next,n-1,Reward(S_next),L,choice)

        if n==2:
            basefunction_mt(S_next,n-1,Reward(S_next),L,ch)


        
        


def mechanical_turk(S):
    possible_actions=PossibleActions(S)
    P=[]
    if (possible_actions==[] or possible_actions==None):
        return []
    Mo_ves=[] 
    Lv=2
    basefunction_mt(S,Lv,0,Mo_ves,0)
    Mo_ves=np.array(Mo_ves)
    # print(Mo_ves)
    for k in range(len(possible_actions)):
        W=np.where(Mo_ves[:,1]==k)
        P.append(np.amin(Mo_ves[W[0],0]))
    P=np.array(P)   
    return P
    





# =============================================================================
# ________________________________________________________
# =============================================================================


