# Student agent: Add your own agent here
from asyncio.windows_events import NULL
from email.policy import default
from inspect import stack
from agents.agent import Agent
from store import register_agent
import numpy as np
import sys
from copy import deepcopy
import time
"""
class Tree:
    def __init__(self, Node=None):
        self.Root = Node
    def getRoot(self):
        return self.Root

class Node:
    def __init__(self,opp,board,mypos,ismax,move=None,parent = None,val = None , child = []):
        self.Child = child #list of nodes
        self.Mypos = mypos #tuple our agent's position
        self.Opp_pos = opp #tuple opponent's position
        self.Board = board #board of current moves
        self.Value = val #int -1 lose 0 tie 1 win
        self.Ismax = ismax #bool max or min true for max false for min
        #!!
        self.Parent = parent # set parent to none once not needed
        self.Move = move,

    def isleaf(self):
        return len(self.Child) == 0
"""

MAX, MIN = 1000, -1000

@register_agent("student_agent")
class StudentAgent(Agent):

    depthg = 5 #default
    num = 50 #default
    stepcount = 0
    uni = np.empty((0, 3), int)
    
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    
    
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.autoplay = True
        #self.Tree = Tree();
        #self.Setup = False  # check if it is setuped


    """
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) #urdl
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.autoplay = True
        self.uni = np.empty((0, 3), int)
    """
    """
    def setup(self, chess_board, my_pos, adv_pos, max_step):
        # level order traversal
        # board???
        # how to check end
        queue = []
        N1 = Node(adv_pos,deepcopy(chess_board),my_pos,True)
        t1 = Tree(N1)
        self.Tree = t1
        queue.append(N1)
        while (len(queue) != 0):
            n = len(queue)
            while (n > 0):
                p = queue[0]
                queue.pop(0) #p is a Node
                check = p.Ismax
                endgame,myscore, opponentscore = self.check_endgame(p.Board, p.Mypos, p.Opp_pos)
                n -= 1
                if(endgame):
                    #print("Here")
                    if(myscore > opponentscore):
                        #print("Win")
                        p.val = 1
                        #win
                    elif(opponentscore > myscore):
                        #print("Lose")
                        p.val = -1
                    else:
                        #print("Tie")
                        p.val = 0
                    continue
                #change board and both pos when creating the Node coresponding to the move
                if(check): #Max mypos and board change
                    uniquemoves = self.uniquemoves(p.Board,p.Mypos,p.Opp_pos,max_step)
                    for umove in uniquemoves:
                        #print(umove)
                        board_copy = deepcopy(p.Board)
                        x,y,dir = umove
                        board_copy[x,y,dir] = True
                        #print(board_copy[:, :, 0])
                        #print(board_copy[:, :, 1])
                        #print(board_copy[:, :, 2])
                        #print(board_copy[:, :, 3])
                        #time.sleep(10000)
                        newNode = Node(p.Opp_pos,board_copy,(x,y),False,p,((x,y),dir))
                        p.Child.append(newNode)
                        queue.append(newNode)
                else:
                    uniquemoves = self.uniquemoves(p.Board,p.Opp_pos,p.Mypos,max_step)
                    for umove in uniquemoves:
                        board_copy = deepcopy(p.Board)
                        x, y, dir = umove
                        board_copy[x,y,dir] = True
                        newNode = Node((x, y), board_copy, p.Mypos , True, p,((x,y),dir))
                        p.Child.append(newNode)
                        queue.append(newNode)
        return
    """
    def ltpo(self,chess_board):
        #adapter
        if chess_board.shape == (5, 5, 4):
            
            self.depthg = 4
            
            self.num = 40
        if chess_board.shape == (6, 6, 4):
            
            self.depthg = 2
            
            self.num = 25
        if chess_board.shape == (7, 7, 4):
            
            self.depthg = 2
            
            self.num = 14
        if chess_board.shape == (8, 8, 4):
            
            self.depthg = 2
            
            self.num = 11
        if chess_board.shape == (9, 9, 4):
            
            self.depthg = 2

            self.num = 6
        if chess_board.shape == (10, 10, 4):
            
            self.depthg = 2
            
            self.num = 5
        if chess_board.shape == (11, 11, 4):
            
            self.depthg = 2
            
            self.num = 4
        if chess_board.shape == (12, 12, 4):
            
            self.depthg = 2
            
            self.num = 3

    #alpha -1000 beta 1000
    def step(self, chess_board, my_pos, adv_pos, max_step):
        #print(self.heuristic_helper(chess_board,my_pos,adv_pos))
        #self.heuristic_computation(chess_board,my_pos,adv_pos)
        #input("here")
        #start_time = time.time()
        bestval = MIN
        bestmove = None

        self.ltpo(chess_board)
        self.stepcount += 1
        print(self.stepcount)
        #print(self.depthg)
        #print(self.num)
        print(chess_board.shape)
        

        uniquemoves = self.ordermoves(chess_board,my_pos,adv_pos,max_step)
        for uniquemove in uniquemoves:
            copy = deepcopy(chess_board)
            x,y,dir = uniquemove
            copy1 = self.set_barrier(x,y,dir,copy)
            val = self.minimax(0,False,copy1,(x,y),adv_pos,max_step,MIN,MAX)
            #print(val)
            if (val == 1):
                return (x,y),dir
            if(val > bestval):
                bestval = val
                bestmove = (x,y),dir
        #print("--- %s seconds ---" % (time.time() - start_time))
        return bestmove
        """
        start_time = time.time()
        self.unique_iterative(chess_board,my_pos,adv_pos,max_step,0)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(self.uni.shape)
        x = self.uniquemoves(chess_board,my_pos,adv_pos,max_step)
        print(x.shape)
        idx = (x[:, None] != self.uni).any(-1).all(1)
        print(x[idx])
        input("PPP")
        """


    def minimax(self, depth, maximizingPlayer, chess_board, my_pos, adv_pos, max_steps, alpha, beta):
        # Terminating condition. i.e
        # leaf node is reached
        chess_board1 = deepcopy(chess_board)
        end, p0_score, p1_score = self.check_endgame(chess_board1,my_pos,adv_pos)
        if end:
            #print("Here")
            return self.whowin(p0_score,p1_score)
        if depth == self.depthg:
            toreturn = self.heuristic_computation(chess_board1, my_pos, adv_pos)
            #print(toreturn)
            return toreturn
        if not end:
            if maximizingPlayer:
                best = -1000
                childs = self.ordermoves(chess_board1, my_pos, adv_pos, max_steps)
                # Recur for left and right children
                for i in childs:
                    x, y,dir = i
                    chess_board1 = deepcopy(chess_board1)
                    new_board = self.set_barrier(x, y, dir, chess_board1)
                    V = self.minimax(depth + 1, False, chess_board1, (x,y), adv_pos, max_steps, alpha, beta)
                    #print(V,depth)
                    best = max(best, V)
                    #print("Max player", best)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
                    #print(best)
                return best
            else:
                best = 1000
                childs = self.ordermoves(chess_board1, adv_pos, my_pos, max_steps)
                for i in childs:
                    x, y,dir = i
                    chess_board1 = deepcopy(chess_board1)
                    new_board = self.set_barrier(x, y, dir, chess_board1)
                    # minimax(self,depth, maximizingPlayer,chess_board,my_pos, adv_pos, max_steps, alpha, beta)
                    V = self.minimax(depth + 1, True, chess_board1, my_pos, (x,y), max_steps, alpha, beta)
                    #print(V,depth)
                    best = min(best, V)
                    #print("Min player", best)
                    beta = min(beta, best)
                    # Alpha Beta Pruning
                    if beta <= alpha:
                        break
                return best


    """
    def unique_iterative(self,chess_board, my_pos, adv_pos, max_steps,cursteps):
        r,c = my_pos
        uniq = np.empty((0, 1), int)
        if cursteps > max_steps:
            return True #base case
        for j in self.dir_map:
            barrier_dir = self.dir_map[j]
            if chess_board[r, c, barrier_dir]:
                continue
            self.uni = np.vstack((self.uni, np.array([r, c, barrier_dir])))

        for i in range(len(self.moves)):
            move = self.moves[i]
            next_pos = tuple(map(lambda i, j: i + j, my_pos, move))
            z,y = next_pos
            if (not chess_board[r,c,i] and next_pos != adv_pos and [z,y] not in self.uni[:, :2].tolist()):
                #print(self.uni[:, :2].tolist())
                #time.sleep(10000)
                #continue
                return np.vstack((uniq, self.unique_iterative(chess_board,next_pos,adv_pos,max_steps,cursteps + 1)))
        
    """
    """
    def unique_iterative(self,chess_board, my_pos, adv_pos, max_steps,cursteps):
        r,c = my_pos
        
        if cursteps > max_steps:
            return  #done
        for j in self.dir_map:
            barrier_dir = self.dir_map[j]
            if chess_board[r, c, barrier_dir] or (any(np.array_equal(x, np.array([r,c,barrier_dir])) for x in self.uni)):
                continue
            self.uni = np.vstack((self.uni, np.array([r, c, barrier_dir])))
        for i in range(len(self.moves)):
            move = self.moves[i]
            next_pos = tuple(map(lambda a, b: a + b, my_pos, move))
            z,y = next_pos
            if (not chess_board[r,c,i] and not next_pos == adv_pos):
                #print(self.uni[:, :2].tolist())
                #time.sleep(10000)
                #continue
                self.unique_iterative(chess_board,next_pos,adv_pos,max_steps,cursteps + 1)
    """

    def whowin(self,p0_score,p1_score):
        if (p0_score > p1_score):
            # print("Win")
            return 1
            # win
        elif (p0_score > p1_score):
            # print("Lose")
            return -1
        else:
            # print("Tie")
            return 0


    
    def ordermoves(self,chess_board, my_pos, adv_pos, max_steps):
        x = self.uniquemoves(chess_board,my_pos,adv_pos,max_steps)
        #print(x.shape)
        np.random.shuffle(x)
        y = x[:self.num]
        return y
    
    """
    def ordermoves(self,chess_board, my_pos, adv_pos, max_steps):
        self.uni = np.empty((0, 3), int)
        self.unique_iterative(chess_board,my_pos,adv_pos,max_steps,0)
        x = deepcopy(self.uni)
        #x = self.uniquemoves(chess_board,my_pos,adv_pos,max_steps)
        np.random.shuffle(x)
        y = x[:20]
        return y
    """

    def randommoves(self):
        return

    def check_reachable(self, chess_board, max_steps, adv_pos, start_pos, check_pos):
        # reusing method in world.
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            # ..
            # print(cur_pos)

            r, c = cur_pos
            if cur_step == max_steps:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = tuple(map(lambda i, j: i + j, cur_pos, move))
                # next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, check_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached

    def uniquemoves(self, chess_board, my_pos, adv_pos, max_steps):
        #start_time = time.time()
        curx, cury = my_pos
        toreturn = np.empty((0, 3), int)
        for steps in range(max_steps + 1):
            if (steps == 0):
                for dir in self.dir_map:
                    if (not chess_board[curx, cury, self.dir_map[dir]]):
                        # print(np.array([checkx,checky2,dir]))
                        toreturn = np.vstack((toreturn, np.array([curx, cury, self.dir_map[dir]])))
            for x in range(steps * -1, steps + 1):  # ...
                y = steps - abs(x)
                checkx = curx + x
                checky = cury + y
                checky2 = cury - y
                check_right = self.check_reachable(chess_board, max_steps, adv_pos, my_pos, np.array([checkx, checky]))
                check_left = self.check_reachable(chess_board, max_steps, adv_pos, my_pos, np.array([checkx, checky2]))
                if (check_left):
                    for dir in self.dir_map:
                        if ((not chess_board[checkx, checky2, self.dir_map[dir]]) and (
                                [checkx, checky2, self.dir_map[dir]]
                                not in toreturn.tolist())):
                            # print(np.array([checkx,checky2,dir]))
                            toreturn = np.vstack((toreturn, np.array([checkx, checky2, self.dir_map[dir]])))
                if (check_right):
                    for dir in self.dir_map:
                        if ((not chess_board[checkx, checky, self.dir_map[dir]]) and (
                                [checkx, checky, self.dir_map[dir]]
                                not in toreturn.tolist())):
                            # print(np.array([checkx, checky2, dir]))
                            toreturn = np.vstack((toreturn, np.array([checkx, checky, self.dir_map[dir]])))
        # print(toreturn)
        #print("--- %s seconds ---" % (time.time() - start_time))
        return toreturn

    stack = NULL
    def uniquemove(self, chess_board, my_pos, adv_pos, cur_step, max_steps):
            #start_time = time.time()
            if cur_step == 0:
                #initialize
                stack = NULL
            if cur_step == max_steps:
                #base case
                return stack
            else:
                curx, cury = my_pos
                for steps in range(max_steps + 1):
                    for x in range(steps * -1, steps + 1):  # ...
                        y = steps - abs(x)
                        checkx = curx + x
                        checky = cury + y
                        checky2 = cury - y
                        check_right = self.check_reachable(chess_board, max_steps, adv_pos, my_pos, np.array([checkx, checky]))
                        check_left = self.check_reachable(chess_board, max_steps, adv_pos, my_pos, np.array([checkx, checky2]))
                        if (check_left):
                            for dir in self.dir_map:
                                if ((not chess_board[checkx, checky2, self.dir_map[dir]]) and ([checkx, checky2, self.dir_map[dir]]
                                        not in stack.tolist())):
                                    # print(np.array([checkx,checky2,dir]))
                                    stack = np.vstack((stack, np.array([checkx, checky2, self.dir_map[dir]])))
                                    return self.uniquemove(self, chess_board, (curx,cury+1), adv_pos, cur_step+1, max_steps)
                        if (check_right):
                            for dir in self.dir_map:
                                if ((not chess_board[checkx, checky, self.dir_map[dir]]) and ([checkx, checky, self.dir_map[dir]]
                                        not in stack.tolist())):
                                    # print(np.array([checkx, checky2, dir]))
                                    stack = np.vstack((stack, np.array([checkx, checky, self.dir_map[dir]])))
                                    return self.uniquemove(self, chess_board, (curx,cury-1), adv_pos, cur_step+1, max_steps)

                # print(toreturn)
        #print("--- %s seconds ---" % (time.time() - start_time))
                

    def check_endgame(self, board, my_pos, adv_pos):
        father = dict()
        board_size = board.shape[0]
        for r in range(board.shape[0]):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        """
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        """
        return True, p0_score, p1_score #p0 is player p1 is opponent

    def set_barrier(self, r, c, dir,board):
        # Set the barrier to True
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        return board

    def covert(self,board):
        sizex = board.shape[0]
        newboardsize = (sizex ) * 2 -1
        up = board[:,:,0]
        r1 = np.where(up == True)
        x,y = r1
        print(x)
        print(y)
        a1 = np.full((newboardsize, newboardsize), False)
        for i in range(len(x)):
            x1 = x[i]
            if x1 == 0:
                continue
            y1 = y[i]
            x2 = x1*2 -1
            y2 = y1*2
            a1[x2,y2] = True
        left = board[:,:,3]
        r2 = np.where(left == True)
        x3,y3 = r2
        print(x3)
        print(y3)
        for j in range(len(x3)):
            y4 = y3[j]
            if y4 == 0:
                continue
            x4 = x3[j]
            x5 = x4*2
            y5 = y4*2 -1
            a1[x5,y5] = True
        print(a1)
        return a1

    #check we are on opponent left/right, up or down depend on how we draw the line
    # only add to down and right
    # winscore / num of ways of ending
    def heuristic_helper(self,board,mypos,advpos):
        #start_time = time.time()
        moves = []
        size = board.shape[0]
        maxbarrier = 0
        smallerx = min(mypos[0],advpos[0])
        biggerx = max(mypos[0],advpos[0])
        #if(biggerx == size-1):
            #biggerx -=1 ;
        for x in range(smallerx,biggerx):
            #print("x is", x)
            down = board[x,:,2]
            #print(down)
            #print(np.count_nonzero(down))
            if (maxbarrier < np.count_nonzero(down)):
                maxbarrier = np.count_nonzero(down)
                moves.clear()
                sublist1 = []
                l = np.where(down == True)
                #print(l)
                l3 = l[0]
                for i in range(size):
                    if i not in l3:
                        sublist1.append((x,i,2))
                moves.append(sublist1)
            elif (maxbarrier == np.count_nonzero(down)):
                sublist = []
                l = np.where(down == True)
                #print(l)
                l3 = l[0]
                for i in range(size):
                    if i not in l3:
                        sublist.append((x,i,2))
                moves.append(sublist)
        #print("max is" , maxbarrier)
        #print(moves)
        smallery = min(mypos[1], advpos[1])
        biggery = max(mypos[1], advpos[1])
        #if(biggery == size-1):
            #biggery -=1 ;
        for y in range(smallery,biggery):
            right = board[:,y,1]
            if(maxbarrier < np.count_nonzero(right)):
                maxbarrier = np.count_nonzero(right)
                moves.clear()
                sublist2 = []
                l = np.where(right == True)
                l4 = l[0]
                for i in range(size):
                    if i not in l4:
                        sublist2.append((i,y,1))
                moves.append(sublist2)
            elif (maxbarrier == np.count_nonzero(right)):
                sublist2 = []
                l = np.where(right == True)
                l4 = l[0]
                for i in range(size):
                    if i not in l4:
                        sublist2.append((i,y,1))
                moves.append(sublist2)
        #print(moves)
        return moves,maxbarrier

    def heuristic_computation(self,board,mypos,advpos):
        moves,maxbarrier = self.heuristic_helper(board,mypos,advpos)
        size = board.shape[0]
        netwin = 0
        game = 0
        for move in moves:
            if(len(move)==0):
                break
            #board = deepcopy(board)
            for barrier in move:
                x,y,dir = barrier
                break
                #board = self.set_barrier(x,y,dir,board)
            if(dir == 2):
                playerup = False
                if mypos[0] < advpos[0]:
                    playerup = True
                up_score = (x+1) * size
                downscore = size ** 2 - up_score
                #print(up_score)
                #print(downscore)
                if(playerup and up_score > downscore):
                    netwin += 1
                elif(playerup and up_score < downscore):
                    netwin -= 1
                elif(not playerup and downscore > up_score):
                    netwin += 1
                elif(not playerup and up_score > downscore):
                    netwin -= 1
                #print("netwin :", netwin)
                #print("Game :" ,game + 1)
            else:
                playerleft = False
                if advpos[1] > mypos[1]:
                    playerleft = True
                rightscore = (y+1) * size
                leftscored = size**2 - rightscore
                #print(rightscore)
                #print(leftscored)
                if(playerleft and rightscore > leftscored):
                    netwin += 1
                elif(playerleft and rightscore > leftscored):
                    netwin -= 1
                elif(not playerleft and leftscored < rightscore): #we at left
                    netwin += 1
                elif(not playerleft and rightscore < leftscored):
                    netwin -= 1
                #print("netwin :", netwin)
                #print("Game :", game + 1)
            game += 1
        if(game == 0):
            #print("board")
            #print(board[:,:,0])
            #print(board[:, :, 1])
            #print(board[:, :, 2])
            #print(board[:, :, 3])
            #print("Wrong")
            end,pos1,pos2 = self.check_endgame(board,mypos,advpos)
            if end:
                return self.whowin(pos1,pos2)
            else: #never used
                #print(moves)
                #print(mypos)
                #print(advpos)
                #print(board[:, :, 0])
                #print(board[:, :, 1])
                #print(board[:, :, 2])
                #print(board[:, :, 3])
                #time.sleep(100000)
                return 0 #game ends
        return netwin/game