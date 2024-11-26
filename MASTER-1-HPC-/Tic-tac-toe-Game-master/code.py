print("***Tic-Tac-Toe Game welcome !!!***")
print("Rules:")
print("1:The game is played on a grid that's 3 squares by 3 squares.\n2:select the number of the case u want to play in\n3:The first player to get 3 of her marks in a row (up, down, across, or diagonally) is the winner.\n4:When all 9 squares are full, the game is over.\n***let's begin***")

def inputControl (xoro) :
    while xoro  not in ("X", "O"):
        print(xoro)
        xoro = input("Please, select only X or O : ")
    return xoro

player1 = input("Player 1 please select your symbol to play(X/O) : ")
player1 = inputControl(player1)

if player1 == "X":
    player2 ="O"
else :
    player2 ="X"

print("Player 1 is playing with",player1)
print("Player 2 is playing with",player2)

print("*******THE GAME START*******")

theBoard = {'7': '7' , '8': '8' , '9': '9' ,
            '4': '4' , '5': '5' , '6': '6' ,
            '1': '1' , '2': '2' , '3': '3' }
'''theBoard un dictionnaire qui va nous permettre d'identifier les cases du tableau du jeu'''         
def printBoard(board):
    print(board['7'] + '  |' + board['8'] + '  |' + board['9'])
    print('---+---+---')
    print(board['4'] + '  |' + board['5'] + '  |' + board['6'])
    print('---+---+---')
    print(board['1'] + '  |' + board['2'] + '  |' + board['3'])

printBoard(theBoard)

def emptyOrnot (position):
    while (theBoard[position] in ("X" , "0") ) :
        position =input("please select an empty position : ")
    return position

turn = 1

for i in range(9) :
    position = input("Player " + str(turn) + ", your turn select the position : ")
    position = emptyOrnot(position)
    if turn == 1 :
        theBoard[position]=player1
        turn+=1
    elif turn == 2 :
        theBoard[position]=player2
        turn-=1
    printBoard(theBoard)
    if i>=4:
        if theBoard['7'] == theBoard['8'] == theBoard['9'] : 
            print("YOU WIN") 
            
            break
        elif theBoard['4'] == theBoard['5'] == theBoard['6'] : 
            print(" YOU WIN") 
            
            break
        elif theBoard['1'] == theBoard['2'] == theBoard['3'] : 
            print(" YOU WIN") 
            
            break
        elif theBoard['1'] == theBoard['4'] == theBoard['7'] : 
            print(" YOU WIN") 
            
            break
        elif theBoard['2'] == theBoard['5'] == theBoard['8'] : 
            print(" YOU WIN") 
            
            break
        elif theBoard['3'] == theBoard['6'] == theBoard['9'] : 
            print(" YOU WIN")
            
            break
        elif theBoard['7'] == theBoard['5'] == theBoard['3'] : 
            print(" YOU WIN") 
            
            break
        elif theBoard['1'] == theBoard['5'] == theBoard['9'] : 
            print(" YOU WIN") 
          
            break
        elif i == 8 :
            print("IT'S A TIE ")

                      
 

