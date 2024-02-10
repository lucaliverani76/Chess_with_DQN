#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:36:53 2023

@author: luca
"""


import numpy as np


#Import custom functions 
import Chessx
import chess
import io
import collections


import cairosvg
from PIL import Image
import cv2
from random import randint



cv2.namedWindow("Chess Board", cv2.WINDOW_NORMAL)


mechturk=False





Player1_white=Chessx.DQNAgent_morebrain(Chessx.params).to(Chessx.params['device'])
# Player1_white.wpath="DQNAgent.pthx"
Player1_white.loadWeights()


Player2_black=Chessx.DQNAgent_morebrain(Chessx.params).to(Chessx.params['device'])
# Player2_black.wpath="DQNAgent.pthx"
Player2_black.loadWeights()


def Convertsvgtonp(svg_data):
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_data))
    return np.array(image)



moves=collections.deque(maxlen=12)# I try to avoi moves repetition
def choosemove(valuesofmoves,possible_actions):
    
    V=np.amax(valuesofmoves)
    W=np.where(valuesofmoves==V)
    move=np.random.choice(W[0]);
    moves.append(possible_actions[move])
    z=set(moves)
    # print(z)
    if len(z)<=4 and len(moves)>11:
        move=randint(0,len(valuesofmoves)-1);
    return(move)

    


def display_match(Model1, Model2, steps=1000):
    
    
    S=Chessx.initialize_sequences()
    
    board = chess.Board()
    
    for n in range(steps):
        
        possible_actions=Chessx.PossibleActions(S)
       
        
        if (possible_actions==[] or possible_actions==None):
            break
        if n%2==0:
            if mechturk and False:
                rew= Chessx.mechanical_turk(S)
            else:
                rew=Chessx.eval_Q_value_generic(Model1,S,possible_actions)
                # print("white")
                # print(rew)
                # print("______")
                TTT=np.vstack((rew[:,0],possible_actions))
                print(TTT.T)


            S=Chessx.Implement_action(S,possible_actions[choosemove(rew,possible_actions)])
            
            print("white")
            print(Chessx.Reward(S)*1000)

        else:
            mirrored_board = Chessx.flip_ranks(board.mirror())
            S_black=Chessx.fromChesstoS(mirrored_board)
            possible_actions=Chessx.PossibleActions(S_black)
            
            if mechturk:
                rew= Chessx.mechanical_turk(S_black)
            else:
                rew=Chessx.eval_Q_value_generic(Model2,S_black,possible_actions)
                # print("black")
                # print(rew)

            # choosemove(rew)
            


            mirrored_board.push_san(possible_actions[choosemove(rew,possible_actions)])    
            re_mirrored_board = Chessx.flip_ranks(mirrored_board.mirror())
            S=Chessx.fromChesstoS(re_mirrored_board)
            print("black")
            print(Chessx.Reward(S)*1000)
            print(S.score[0]*1000)
            print("_________")
            

            
        

        
        # Apply the move to the board
        board.set_fen(S.loc[0,"board"])
        # print(board.is_check())

        # Generate SVG for the updated board position
        svg_board = chess.svg.board(board=board, size=1300)
        
        open_cv_image=Convertsvgtonp(svg_board)

        
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        
        cv2.imshow("Chess Board", open_cv_image)
        
        cv2.waitKey(1000)  # Adjust the delay (in milliseconds)

        # x=input()
        
        # print("ÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖ")
        # print("ÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖ")
        # print("ÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖ")



display_match(Player1_white, Player2_black)


sample_layer_weights = Player1_white.state_dict()

# # Create a chess board
# board = chess.Board()

# # Make some moves on the original board
# board.push(chess.Move.from_uci("e2e4"))
# board.push(chess.Move.from_uci("e7e5"))
# board.push(chess.Move.from_uci("g1f3"))





# # Print the original board after moves
# print("Original Board after moves:")
# print(board)

# # Create a mirrored board
# mirrored_board = flip_ranks(board.mirror())


# # Print the mirrored board
# print("\nMirrored Board:")
# print(mirrored_board)