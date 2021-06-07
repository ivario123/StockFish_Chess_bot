import random
import keyboard
from stockfish import Stockfish
import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
import time
stockfish = Stockfish(r'--- Path to you stockfísh exe ---')

    
    
    
    
    
    
    
    
import numpy as np
import pyautogui
import cv2
import math
import keyboard
from random import sample, shuffle
from math import sqrt


def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def move(OldX, OldY, NewX, NewY):
    OldY = 7-OldY
    NewY = 7-NewY
    print("Trying to click at : {},{} and {},{}".format(OldX, OldY, NewX, NewY))
    if (Side == "White")==Move:
        
        pyautogui.click(x1+remap(OldX,0,8,0,x2-x1)+random.randint(-20,20)+RektSize/2,y1+random.randint(-20,20)+remap(OldY,0,8,0,y2-y1)+RektSize/2)
        time.sleep(random.uniform(0.1,3))
        pyautogui.click(x1+remap(NewX,0,8,0,x2-x1)+random.randint(-20,20)+RektSize/2,y1+random.randint(-20,20)+remap(NewY,0,8,0,y2-y1)+RektSize/2)
    


Template = cv2.imread("template.png", 0)
TW, TH = Template.shape[::-1]
Colors = ["White", "Black"]
Pieces = {
    "Farmer": 1,
    "Runner": 6,
    "Queen": 9,
    "Horse": 3,
    "Tower": 5,
    "King": 1000
}
PieceImg = {}
def makemove(side = "White",lastScore = float("-inf"),score = -4,TempBoard = np.array((8,8)),Player = ((2,2),(2,2)),px = 0,py= 0,i = 0,depth = 3):
                        lastScore = score
                        if side == "White":
                            return  (((Player[1][1], Player[1][0]), (px, py),
                                              Pieces["Farmer"]*(score-SimMove("Black", TempBoard, depth, i,lastScore,score))))
                        else:
                            return (((Player[1][1], Player[1][0]), (px, py),
                                              Pieces["Farmer"]*(score-SimMove("White", TempBoard, depth, i,lastScore,score))))

def LoadPieces():

    for color in Colors:
        for piece in list(Pieces.keys()):
            im = cv2.imread(
                "{}{}.png".format(piece, color), 0)
            fac = (int(im.shape[1]*(x2-x1)/TW), int(im.shape[0]*(y2-y1)/(TH)))
            PieceImg["{}{}".format(piece, color)] = cv2.resize(im, fac)


board = []

fenNot = {
    "Black King" : "k",
    "White King" : "K",
    "Black Queen" : "q",
    "White Queen" : "Q",
    "Black Runner" : "b",
    "White Runner" : "B",
    "Black Horse" : "n",
    "White Horse" : "N",
    "Black Tower" : "r",
    "Black Farmer" : "p",
    "White Farmer" : "P",
    "White Tower" : "R"
}
def ReadBoard(Side):
    board = []
    for i in range(8):
        board.append([])
        for j in range(8):
            board[i].append("")
    imageColor = pyautogui.screenshot()
    imageColor = cv2.cvtColor(np.array(imageColor), cv2.COLOR_RGB2BGR)
    imageColor = cv2.cvtColor(np.array(imageColor), cv2.COLOR_BGR2RGB)
    imageColor = imageColor[y1:y2, x1:x2]
    imageColor[np.all(imageColor == (118, 150, 86), axis=-1)] = (238, 238, 210)
    imageColor[np.all(imageColor == (186, 202, 43), axis=-1)] = (238, 238, 210)
    
    image = cv2.cvtColor(np.array(imageColor), cv2.COLOR_BGR2GRAY)
    for color in Colors:
        for piece in list(Pieces.keys()):
            s = piece+color
            w, h = PieceImg[s].shape[::-1]
            res = cv2.matchTemplate(image, PieceImg[s], cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                x = int(math.floor(remap(pt[0], 0, x2-x1, 0, 8)))
                y = int(math.floor(remap(pt[1]+10, 0, y2-y1, 0, 8)))
                board[y][x] = "{} {}".format(color, piece)
                if(color == "White"):
                    cv2.rectangle(
                        imageColor, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(
                        imageColor, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)
    cv2.imwrite("in_memory_to_disk.png", imageColor)
    #if(board!=OldBoard):
    fen = ""
    board = np.array(board)
    return board
    #bestX = int(remap(best[0], 0, 8, 0, x2-x1))
    #bestY = int(remap(best[1], 0, 8, 0, y2-y1))
    #s_img = cv2.cvtColor(PieceImg[char+Side], cv2.COLOR_GRAY2RGB)
    #imageColor[bestX:bestX+s_img.shape[0], bestY:bestY+s_img.shape[1]] = s_img
    #cv2.rectangle(
    #                    imageColor, pt, (bestX + 10, bestY + 10), (0, 0, 255), 2)
    #cv2.imshow("Screenshot", imutils.resize(imageColor, width=600))
    #cv2.waitKey(0)

def toFen(board):
    fen = ""
    y,x = board.shape
    WhiteCastlePossible = True
    BlackCastlePosible = True
    for Y in range(0,y,1):
        counter = 0
        for X in range(0,x,1):
            if board[Y][X] == "":
                counter+=1
            else:
                if counter!=0:
                    fen+=str(counter)
                    counter = 0
                fen+=fenNot[board[Y][X]]
        if counter!=0:
            fen+=str(counter)
        if Y!=y-1:
            fen+= '/'
    fen+= " w "if Move else " b "
    fen+="- "
    fen+="- 0 0" 
    
    return fen
Side = ""

letToNum = {
    'a' : 0,
    'b' : 1,
    'c' : 2,
    'd' : 3,
    'e' : 4,
    'f' : 5,
    'g' : 6,
    'h' : 7
}

Move = True
if __name__ == "__main__":
    for i in range(8):
        board.append([])
        for j in range(8):
            board[i].append("")
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('1'):  # if key 'q' is pressed
                x1, y1 = pyautogui.position()
                print("(x,y) = ({},{})".format(x1, y1))
                break  # finishing the loop
        except:
            continue
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('2'):  # if key 'q' is pressed
                x2, y2 = pyautogui.position()
                print("(x,y) = ({},{})".format(x2, y2))
                break  # finishing the loop
        except:
            continue
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('w'):  # if key 'q' is pressed
                Side = "White"

                break  # finishing the loop
            if keyboard.is_pressed('b'):  # if key 'q' is pressed
                Side = "Black"
                break  # finishing the loop
        except:
            continue
    RektSize = (y2-y1)/8
    print("Side = {}".format(Side))
    LoadPieces()
    NewBoard = ReadBoard(Side)
    LastBoard= NewBoard
    Move = True if Side == "White" else False
    first = True
    LastSide = Move
    counter = 0 
    while 1:
        try:
            NewBoard = ReadBoard(Side)
            if counter > random.randint(10,30):
                Move = True if Side == "White" else False
                first = True
                print("Reseting")
            if NewBoard.shape == LastBoard.shape:
                counter += 1
                print(("White" if Move else "Black"))
                if first or ((NewBoard != LastBoard).any() and Move == (Side=="White")):
                    counter = 0
                    LastMove = Move
                    
                    first = False
                    stockfish.set_fen_position(toFen(NewBoard))
                    print(stockfish.get_board_visual())
                    print("White" if Move else "Black")
                    if (Side == "White")==Move:
                        best = stockfish.get_best_move_time(10)
                        print(best)
                    move1 = (0,0)
                    move2 = (0,0)
                    move(letToNum[best[0]], int(best[1])-1, letToNum[best[2]], int(best[3])-1)
                    for y in range(0,7):
                        for x in range(0,7):
                            if NewBoard[y][x]!="" and LastBoard[y][x]!= NewBoard[y][x]:
                                Move = False if NewBoard[y][x].find("White") else True
                                    
                    LastBoard= NewBoard
        except KeyboardInterrupt:
                    Move = not Move
                    first = True
                
