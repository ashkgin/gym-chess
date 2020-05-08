#!/usr/bin/env python
# coding: utf-8

######### state #######################
#

"""
Design Details

- In any state it is white's move except in case of check mate and draw
- There are boards, 1 - position board, 2 - analysis board
- position board - it tells the position of pieces
- analysis board - it tells analysis of a particular position board
- with each move position and analysis board change

initial state is a list of size rows x cols

Features:

For the purpose of visualization and debugging, following features are added

1. conversion of action into alpha numeric move
2. Display of board position
3. Dumping of board position in a file
4. configurable board size for rows and columns with minimum 3 and maximum 8
"""

# # Chess
import sys
from contextlib import closing

import numpy as np
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import seaborn as sns
import copy

BLANK_SQ_ID=0
WHITE_KING_ID=1
WHITE_ROOK_ID=2
BLACK_KING_ID=4
INITIAL_BOARD = {
    "3x3": [
        [BLANK_SQ_ID, BLANK_SQ_ID, WHITE_ROOK_ID],
        [BLANK_SQ_ID, BLANK_SQ_ID, WHITE_KING_ID],
        [BLACK_KING_ID, BLANK_SQ_ID, BLANK_SQ_ID],
    ],
    "4x4": [
            [BLANK_SQ_ID,      BLANK_SQ_ID, WHITE_ROOK_ID,   BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, WHITE_KING_ID,   BLANK_SQ_ID],
            [BLACK_KING_ID,    BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID]
            ],
    "8x8": [
            [BLANK_SQ_ID,      BLANK_SQ_ID, WHITE_ROOK_ID,   BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, WHITE_KING_ID,   BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLACK_KING_ID,    BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID],
            [BLANK_SQ_ID,      BLANK_SQ_ID, BLANK_SQ_ID,     BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID, BLANK_SQ_ID]
            ],
}

class ChessEnv(discrete.DiscreteEnv):
    """Chess environment."""
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, board_size=(4,4)):

        self.state_cnt=0

        self.glb_board_size_row = board_size[0]
        self.glb_board_size_col = board_size[1]

        assert self.glb_board_size_row <= 8 and self.glb_board_size_row >= 3, "board size must be between 3 and 8"
        assert self.glb_board_size_col <= 8 and self.glb_board_size_col >= 3, "board size must be between 3 and 8"

        #board display settings
        rows = self.glb_board_size_row
        cols = self.glb_board_size_col
        rows_nums = ['8', '7', '6', '5', '4', '3', '2', '1']
        cols_alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.glb_cols_alpha = [cols_alpha[i] for i in range(cols)]
        self.glb_rows_nums = [rows_nums[i+8-rows] for i in range(rows)]

        self.xticks = [self.glb_cols_alpha[i] for i in range(cols)]
        self.yticks = [self.glb_rows_nums[i] for i in range(rows)]

        #position board IDs
        self.glb_blank_sq_id        = BLANK_SQ_ID
        self.glb_white_king_id      = WHITE_KING_ID
        self.glb_white_rook_id      = WHITE_ROOK_ID
        self.glb_black_king_id      = BLACK_KING_ID
        self.glb_num_king_moves     = 8
        
        #analysis board IDs
        self.glb_white_to_move      = 1
        self.glb_white_king_sq      = 1
        self.glb_white_rook_sq      = 2
        self.glb_white_king_access  = 4
        self.glb_white_rook_access  = 8
        
        self.glb_black_king_sq      = 256
        self.glb_black_king_access  = 512

        if board_size==(3,3):
            self.glb_initial_state = INITIAL_BOARD["3x3"]
        elif board_size==(4,4):
            self.glb_initial_state = INITIAL_BOARD["4x4"]
        elif board_size==(8,8):        #currently 8x8 board is not supported
            self.glb_initial_state = INITIAL_BOARD["8x8"]
        else:
            board_size = (4, 4)
            self.glb_initial_state = INITIAL_BOARD["4x4"]

        self.glb_action_list = []

        #append king's move
        self.glb_action_list.append([-1, -1])
        self.glb_action_list.append([-1, 0])
        self.glb_action_list.append([-1, 1])
        self.glb_action_list.append([0, -1])
        self.glb_action_list.append([0, 1])
        self.glb_action_list.append([1, -1])
        self.glb_action_list.append([1, 0])
        self.glb_action_list.append([1, 1])

        #append rook's move, rook can move to four directions
        dirs = [[-1, 0],        #left
                [1, 0],         #right
                [0, -1],        #up
                [0, 1]          #down
                ]

        self.glb_rook_dirs = dirs

        for i in range(4):          
            if i < 2:
                for j in range (1, self.glb_board_size_row):
                    r = dirs[i][0]
                    c = dirs[i][1]
                    self.glb_action_list.append([j*r, 0])
            else:
                for j in range (1, self.glb_board_size_col):
                    r = dirs[i][0]
                    c = dirs[i][1]
                    self.glb_action_list.append([0, j*c])
    
        self.num_rook_moves = len(self.glb_action_list) - self.glb_num_king_moves

        #display chess board position
        self.display_chess_board(self.glb_initial_state)
        self.all_states = self.generate_all_states(self.glb_initial_state)

        # Create inverse mapping of states
        self.state_mapping = {}
        self.inverse_mapping = {}
        for i in range(len(self.all_states)):
            tup = tuple(map(tuple, self.all_states[i]))
            self.state_mapping[i] = tup
            self.inverse_mapping[tup] = i

        self.nS = len(self.all_states)
        self.nA = len(self.glb_action_list)

        print('total number of states = ', self.nS)

        ## Generating probability matrix
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        self.generate_probs(self.glb_initial_state, self.P)

        self.isd = np.array([1 for s in range(len(self.all_states))]).astype('float').ravel()
        self.isd /= self.isd.sum()

        super(ChessEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def get_analysis_state(self, st, white_to_move=1):
        """ create analysis
        'Analysis state' tells following
        1. check
        2. mate
        3. stalemate
        4. position of each piece
        5. square accessible by each piece
        """

        is_check = 0
        is_mate = 0
        is_stalemate = 0

        anaState = np.zeros((self.glb_board_size_row, self.glb_board_size_col))

        #set white king sq
        r_wh_K, c_wh_K = self.get_piece_sq(st, self.glb_white_king_id)
        anaState[r_wh_K, c_wh_K] += self.glb_white_king_sq

        # set white rook sq
        r_wh_R, c_wh_R = self.get_piece_sq(st, self.glb_white_rook_id)
        anaState[r_wh_R, c_wh_R] += self.glb_white_rook_sq

        #set black king sq
        r_bl_K, c_bl_K = self.get_piece_sq(st, self.glb_black_king_id)
        anaState[r_bl_K, c_bl_K] += self.glb_black_king_sq

        #add white rook access
        rook_steps = self.get_rook_steps(st, self.glb_white_rook_id)
        for step in rook_steps:
            r = step[0]
            c = step[1]
            anaState[r, c] += self.glb_white_rook_access

        #add king access
        king_steps, list_idx = self.get_king_steps(st, white_to_move)
        if white_to_move==1:
            # search white king steps

            for step in king_steps:
                r = step[0]
                c = step[1]
                anaState[r, c] += self.glb_white_king_access

        else:
            # search black king steps
            is_check = self.is_other_rook_access(self, st, r_bl_K, c_bl_K, r_wh_R, c_wh_R)
            len_king_steps = len(king_steps)
            if len_king_steps == 0:
                if is_check:
                    is_mate = 1
                else:
                    is_stalemate = 1

            for step in king_steps:
                r = step[0]
                c = step[1]
                anaState[r, c] += self.glb_black_king_access

        return anaState, is_check, is_mate, is_stalemate

    def is_other_king_access(self, rk1, ck1, rk2, ck2):
        """Checks if other king access same sqaure"""
        r_diff = abs(rk1 - rk2)
        c_diff = abs(ck1-ck2)

        if r_diff <= 1 and c_diff <= 1:
            return True
        else:
            return False

    def is_other_rook_access(self, st, r, c, rR, cR):
        """Checks if other king access same sqaure"""

        if r != rR and c != cR:   #if neither row match nor column
            return False

        if r==rR and c==cR:     #if it is rook's square return False so that black king could capture it
            return False

        if r == rR:
            if c > cR:
                for i in range(1, c-cR):
                    if st[r][cR+i] != 0:
                        return False            #if piece in between return false
            else:
                for i in range(1, c-cR):
                    if st[r][c+i] != 0:
                        return False            #if piece in between return false

        if c == cR:
            if r > rR:
                for i in range(1, r-rR):
                    if st[rR+i][c] != 0:
                        return False            #if piece in between return false
            else:
                for i in range(1, r-rR):
                    if st[r+i][c] != 0:
                        return False            #if piece in between return false

        return True

    def is_state_valid(self, st):
        """
        Checks if a state is valid.
        1. State is invalid if it white's move and black king is in check
        """
        # set white king sq
        r_wh_K, c_wh_K = self.get_piece_sq(st, self.glb_white_king_id)
        # set white rook sq
        r_wh_R, c_wh_R = self.get_piece_sq(st, self.glb_white_rook_id)
        # set black king sq
        r_bl_K, c_bl_K = self.get_piece_sq(st, self.glb_black_king_id)

        is_check = self.is_other_rook_access( st, r_bl_K, c_bl_K, r_wh_R, c_wh_R)
        if is_check:
            return False

        return True

    def get_black_state(self, st):
        """
        Checks the state of black's king
        0. move's exist = 0
        1. check-mate = 1
        2. stale-mate = 2
        """

        white_to_move = 0
        state = 0
        # set white rook sq
        r_wh_R, c_wh_R = self.get_piece_sq(st, self.glb_white_rook_id)
        # set black king sq
        r_bl_K, c_bl_K = self.get_piece_sq(st, self.glb_black_king_id)

        is_check = self.is_other_rook_access( st, r_bl_K, c_bl_K, r_wh_R, c_wh_R)

        king_steps, list_idx = self.get_king_steps(st, white_to_move)
        num_moves = len(king_steps)
        if num_moves==0:
            if is_check == True:
                state = 1            #check mate
            else:
                state = 2            #stale mate
        else:
            state = 0                #moves available

        return state, king_steps, list_idx

    def are_kings_left(self, st):
        """
        Checks if only kings left
        """
        for r in range(self.glb_board_size_row):
            for c in range(self.glb_board_size_col):
                if st[r][c] == self.glb_white_rook_id:
                    return False
        return True

    def is_game_over(self, st):
        """
        Checks if only kings left
        """
        result = self.are_kings_left(st)
        if result:
            return True

        for r in range(self.glb_board_size_row):
            for c in range(self.glb_board_size_col):
                if st[r][c] == -1 or st[r][c] == -2:
                    return True

        return False

    def sq_in_board(self, r, c):
        """check if the square is within the board"""
        if r >= 0 and r<self.glb_board_size_row and c >= 0 and c < self.glb_board_size_col:
            return True
        else:
            return False

    #pID piece id
    def get_piece_sq(self, st, id):
        """get the square where piece is present"""

        for r in range(self.glb_board_size_row):
            for c in range(self.glb_board_size_col):
                if st[r][c] == id:
                    return r, c
        return r, c

    def is_white_rook_supported(self, st, rid):
        """check if white rook is supported white king"""

        r_wh_K, c_wh_K = self.get_piece_sq(st, self.glb_white_king_id)
        # set white rook sq
        r_wh_R, c_wh_R = self.get_piece_sq(st, rid)

        r_diff = abs(r_wh_K - r_wh_R)
        c_diff = abs(c_wh_K - c_wh_R)

        if r_diff <= 1 and c_diff <= 1:
            return True
        else:
            return False

    def get_king_steps(self, st, white_to_move):
        """get the list of sqaures in king's access"""
        king_step_list=[]
        king_step_list_idx = []

        r_wh_K, c_wh_K = self.get_piece_sq(st, self.glb_white_king_id)
        # set white rook sq
        r_wh_R, c_wh_R = self.get_piece_sq(st, self.glb_white_rook_id)

        #set black king sq
        r_bl_K, c_bl_K = self.get_piece_sq(st, self.glb_black_king_id)

        if white_to_move==1:
            # search king steps
            for i in range(self.glb_num_king_moves):
                step = self.glb_action_list[i]
                r = r_wh_K + step[0]  # get row of next step
                c = c_wh_K + step[1]  # get col of next step
                in_board = self.sq_in_board(r, c)

                if in_board == True and st[r][c] == 0:  # square is in board and there is no other piece
                    is_other_king = self.is_other_king_access(r, c, r_bl_K, c_bl_K)        #check if it violate other king's region
                    if not is_other_king:
                        #king_step_list.append([r, c])
                        king_step_list.append([step[0], step[1]])
                        king_step_list_idx.append(i)
                    else:
                        king_step_list.append([0, 0])
                        king_step_list_idx.append(i)
                else:
                    king_step_list.append([0, 0])
                    king_step_list_idx.append(i)

        else:
            # search king steps
            for i in range(self.glb_num_king_moves):
                step = self.glb_action_list[i]
                r = r_bl_K + step[0]  # get row of next step
                c = c_bl_K + step[1]  # get col of next step
                in_board = self.sq_in_board(r, c)

                is_support = self.is_white_rook_supported(st, self.glb_white_rook_id) #rook id because there could be two rook in upgrade problem
                if is_support:
                    if in_board == True and st[r][c] == 0:  # square is in board and there is no other piece
                        is_other_king = self.is_other_king_access(r, c, r_wh_K, c_wh_K)        #check if it violate other king's region
                        is_other_rook = self.is_other_rook_access(st, r, c, r_wh_R, c_wh_R)  # check if it violate other king's region
                        if is_other_king==0 and is_other_rook==0:
                            king_step_list.append([step[0], step[1]])
                            king_step_list_idx.append(i)
#                        else:
#                            king_step_list.append([0, 0])
#                            king_step_list_idx.append(i)
#                    else:
#                        king_step_list.append([0, 0])
#                        king_step_list_idx.append(i)
                else:
                    if in_board == True and st[r][c] != 1:  # square is in board and there is no white king on the square
                        is_other_king = self.is_other_king_access(r, c, r_wh_K, c_wh_K)        #check if it violate other king's region
                        is_other_rook = self.is_other_rook_access(st, r, c, r_wh_R, c_wh_R)  # check if it violate other rook's region
                        if is_other_king==0 and is_other_rook==0:
                            king_step_list.append([step[0], step[1]])
                            king_step_list_idx.append(i)
#                        else:
#                            king_step_list.append([0, 0])
#                            king_step_list_idx.append(i)
#                   else:
#                       king_step_list.append([0, 0])
#                       king_step_list_idx.append(i)

        return king_step_list, king_step_list_idx

    def get_rook_steps(self, st, id):
        """get the list of squares in rook's access"""

        rookStartIdx = self.glb_num_king_moves
        rook_step_list=[]
        rook_step_list_idx=[]
        rR, cR = self.get_piece_sq(st, id)

        dirs = self.glb_rook_dirs
        for i in range(4):
            if i < 2:
                idx = rookStartIdx + i*(self.glb_board_size_row-1)
                for j in range (1, self.glb_board_size_row):
                    r = j*dirs[i][0]        #get distance from target square
                    c = j*dirs[i][1]

                    rNew = rR + r
                    cNew = cR + c

                    in_board = self.sq_in_board(rNew, cNew)
                    if in_board == True and st[rNew][cNew] == 0:  # square is in board and there is no other piece
                        rook_step_list.append([r, c])
                        rook_step_list_idx.append(idx)
                        idx+=1
                    else:
                        rook_step_list.append([0, 0])
                        rook_step_list_idx.append(idx)
                        idx += 1

            else:
                idx = rookStartIdx + 2*(self.glb_board_size_row-1) +(i-2) * (self.glb_board_size_col-1)
                for j in range (1, self.glb_board_size_col):
                    r = j*dirs[i][0]
                    c = j*dirs[i][1]
                    rNew = rR + r
                    cNew = cR + c
                    in_board = self.sq_in_board(rNew, cNew)
                    if in_board == True and st[rNew][cNew] == 0:  # square is in board and there is no other piece
                        rook_step_list.append([r, c])
                        rook_step_list_idx.append(idx)
                        idx += 1
                    else:
                        rook_step_list.append([0, 0])
                        rook_step_list_idx.append(idx)
                        idx += 1

        return rook_step_list, rook_step_list_idx

    def update_board(self, st, id, move):
        """
        Update the board as per move and return new board position
        """

        st_new = copy.deepcopy(st)

        sq = self.get_piece_sq(st_new, id)
        r = sq[0]
        c = sq[1]
        st_new[r][c] = self.glb_blank_sq_id         #empty the square from where piece is moved

        r = sq[0] + move[0]
        c = sq[1] + move[1]

        st_new[r][c] = id                           #update the sqaure where piece is moved

        #display the new board position
        #self.display_chess_board(st_new)
        #self.dump_state_in_file(st_new)

        #return new state
        return st_new

    def list_of_states_from_white_king_move(self, st, P=None):
        """
        Generate all the states for King's move in st
        """

        states = []

        #get white king's move
        white_to_move = 1
        k_steps, idx_list = self.get_king_steps(st, white_to_move)
        iCnt=0          #idx counter for white's move
        for step in k_steps:
            if step[0] != 0 or step[1]!=0:
                new_black_st = self.update_board(st, self.glb_white_king_id, step)

                idx = idx_list[iCnt]
                iCnt += 1
                if P!=None:
                    PLIST = self.P[self.state_cnt][idx]

                #get states after black's move
                # get black king's move
                white_to_move = 0

                black_state, black_king_steps, list_idx = self.get_black_state(new_black_st)
                len_black_king_steps = len(black_king_steps)
                if len_black_king_steps != 0:
                    prob = 1. / len_black_king_steps
                if black_state == 0:  # moves avaible
                    # black_king_steps = self.get_king_steps(new_black_st, white_to_move)
                    for black_step in black_king_steps:
                        new_white_st = self.update_board(new_black_st, self.glb_black_king_id, black_step)
                        states.append(new_white_st)
                        if P!=None:
                            PLIST.append([prob, self.inverse_mapping[tuple(map(tuple, new_white_st))], 0, False])
                else:
                    r, c = self.get_piece_sq(new_black_st, self.glb_black_king_id)
                    if black_state == 1:
                        new_black_st[r][c] = -1
                        if P!=None:
                            PLIST.append([1., self.inverse_mapping[tuple(map(tuple, new_black_st))], 1., True])
                    else:
                        new_black_st[r][c] = -2
                        if P!=None:
                            PLIST.append([1., self.inverse_mapping[tuple(map(tuple, new_black_st))], -1., True])
                    states.append(new_black_st)  # add check mate or stale mate
            else:
                idx = idx_list[iCnt]
                iCnt += 1
                if P != None:
                    PLIST = self.P[self.state_cnt][idx]
                    PLIST.append([1., self.inverse_mapping[tuple(map(tuple, st))], 0, False])

        return states

    def list_of_states_from_white_rook_move(self, st, P=None):
        """
        Generate all the states for King's move in st
        """

        states = []

        #get white rook's move
        white_to_move = 1
        r_steps, idx_list = self.get_rook_steps(st, self.glb_white_rook_id)
        iCnt=0          #idx counter for white's move
        for step in r_steps:
            if step[0] !=0 or step[1]!=0:
                new_black_st = self.update_board(st, self.glb_white_rook_id, step)

                idx = idx_list[iCnt]
                iCnt += 1
                if P!=None:
                    PLIST = self.P[self.state_cnt][idx]

                #get states after black's move
                # get black king's move
                white_to_move = 0
                black_state, black_king_steps, list_idx = self.get_black_state(new_black_st)
                len_black_king_steps = len(black_king_steps)
                if len_black_king_steps != 0:
                    prob = 1. / len_black_king_steps
                if black_state==0:      #moves avaible
                    #black_king_steps = self.get_king_steps(new_black_st, white_to_move)
                    for black_step in black_king_steps:
                        new_white_st = self.update_board(new_black_st, self.glb_black_king_id, black_step)
                        states.append(new_white_st)
                        rew = 0
                        is_draw = self.are_kings_left(st)
                        if is_draw:
                            rew = -1.
                            r, c = self.get_piece_sq(new_white_st, self.glb_black_king_id)
                            new_white_st[r][c] = -2
                        if P!=None:
                            PLIST.append([prob, self.inverse_mapping[tuple(map(tuple, new_white_st))], rew, False])
                else:
                    r, c = self.get_piece_sq(new_black_st, self.glb_black_king_id)
                    if black_state == 1:
                        new_black_st[r][c] = -1
                        if P!=None:
                            PLIST.append([1., self.inverse_mapping[tuple(map(tuple, new_black_st))], 1., True])
                    else:
                        new_black_st[r][c] = -2
                        if P!=None:
                            PLIST.append([1., self.inverse_mapping[tuple(map(tuple, new_black_st))], -1., True])
                    states.append(new_black_st)      #add check mate or stale mate
            else:
                idx = idx_list[iCnt]
                iCnt += 1
                if P != None:
                    PLIST = self.P[self.state_cnt][idx]
                    PLIST.append([1., self.inverse_mapping[tuple(map(tuple, st))], 0, False])

        return states

    def generate_all_states(self, initial_state):
        """
        Generate all the states for MDPs

        keep adding the states in states_search and get the states from states_search
        incrementally and when there are no more new states left stop

        """
        cnt = 0
        states_search = []
        states_search.append(initial_state)
        #self.dump_state_in_file(initial_state, cnt)
        #self.display_chess_board(initial_state, cnt)
        cnt += 1
        i=0
        while True:
            print(i)
            if i == len(states_search):
                break
            st = states_search[i]
            is_game_over = self.is_game_over(st)
            if is_game_over==False:
                states = self.list_of_states_from_white_king_move(st)
                for new_st in states:
                    if new_st not in states_search:
                        states_search.append(new_st)
                        #self.dump_state_in_file(new_st, cnt)
                        #self.display_chess_board(new_st, cnt)
                        cnt += 1

                #find new states arised from rook's move
                states = self.list_of_states_from_white_rook_move(st)
                for new_st in states:
                    if new_st not in states_search:
                        states_search.append(new_st)
                        #self.dump_state_in_file(new_st, cnt)
                        #self.display_chess_board(new_st, cnt)
                        cnt += 1
            i += 1

        return states_search

    def generate_probs(self, initial_state, P):
        """
        Generate all the states probability matrix

        keep adding the states in states_search and get the states from states_search
        incrementally and when there are no more new states left stop

        """
        cnt = 0
        states_search = []
        states_search.append(initial_state)
        #self.dump_state_in_file(initial_state, cnt)
        #self.display_chess_board(initial_state, cnt)
        cnt += 1
        i=0

        while True:
            if i == len(states_search):
                break
            st = states_search[i]
            is_game_over = self.is_game_over(st)
            if is_game_over==False:
                states = self.list_of_states_from_white_king_move(st, P=P)
                for new_st in states:
                    if new_st not in states_search:
                        states_search.append(new_st)
                        #self.dump_state_in_file(new_st, cnt)
                        #self.display_chess_board(new_st, cnt)
                        cnt += 1

                #find new states arised from rook's move
                states = self.list_of_states_from_white_rook_move(st, P=P)
                for new_st in states:
                    if new_st not in states_search:
                        states_search.append(new_st)
                        #self.dump_state_in_file(new_st, cnt)
                        #self.display_chess_board(new_st, cnt)
                        cnt += 1
            else:
                for k in range(len(self.glb_action_list)):
                    PLIST = P[self.state_cnt][k]
                    PLIST.append([1., self.inverse_mapping[tuple(map(tuple, st))], 0, True])
                
            self.state_cnt += 1
            i += 1

        return states_search

    def get_alphaNum_move(self, st, move_num):
        """convert move to alpha numeric move"""

        if move_num < self.glb_num_king_moves:             #these are king moves
            sq = self.get_piece_sq(st, self.glb_white_king_id)
            step = self.glb_action_list[move_num]
            r = sq[0] + step[0]
            c = sq[1] + step[1]

            alpha = self.glb_cols_alpha[c]
            num = str(self.glb_rows_nums[r])
            move = 'K'+alpha+num
        else:
            sq = self.get_piece_sq(st, self.glb_white_rook_id)
            step = self.glb_action_list[move_num]
            r = sq[0] + step[0]
            c = sq[1] + step[1]

            alpha = self.glb_cols_alpha[c]
            num = str(self.glb_rows_nums[r])
            move = 'R'+alpha+num

        return move

    def display_chess_board(self, board_state, state_num=None, title='Chess Board'):
        """Display the chess board position"""
        #time = datetime.datetime.now().strftime("%I%M%d%Y")
        rows = self.glb_board_size_row
        cols = self.glb_board_size_col
        xticks = self.xticks
        yticks = self.yticks
        if state_num!=None:
            title = title + ' state ' + str(state_num)

        plt.figure(figsize=(4, 4))
        plt.title(title)

        policy_labels = np.empty([rows, cols], dtype='<U10')
        disp_board = np.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                if row & 1:
                    if col & 1:
                        disp_board[row, col] = 3
                    else:
                        disp_board[row, col] = 0
                else:
                    if col & 1:
                        disp_board[row, col] = 0
                    else:
                        disp_board[row, col] = 3

                if board_state[row][col] == 1:
                    policy_labels[row][col] = 'Kw'

                if board_state[row][col] == 2:
                    policy_labels[row, col] = 'Rw'

                if board_state[row][col] == 4:
                    policy_labels[row, col] = 'Kb'

                if board_state[row][col] == -1:
                    policy_labels[row, col] = 'Kbm'

                if board_state[row][col] == -2:
                    policy_labels[row, col] = 'Kbd'


        sns.heatmap(disp_board, annot=policy_labels, xticklabels=xticks, yticklabels=yticks, fmt='', linewidths=.5,
                    cbar=False)
        plt.gcf()
        plt.savefig('states' + '/' + title )
        #plt.show()
        plt.close()

    def dump_state_in_file(self, st, state_num=None):
        """dump state in a file"""
        logFile = 'chess_logfile.txt'

        with open(logFile, "a") as hFile:
            hFile.write("\nchess state ")
            if state_num!=None:
                hFile.write(str(state_num))
            hFile.write("\n\n")
            for s in st:
                hFile.write(str(s))
                hFile.write("\n")
            hFile.write("\n\n")
            hFile.close()