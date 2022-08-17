from pprint import pprint


def find_empty_space(sudoku):
    

   
    for r in range(9):
        for c in range(9):
            if sudoku[r][c] == 0:
                return r, c

    return None, None  # if no spaces in the puzzle are empty (-1)

def validity(sudoku, guess, row, col):
   
    # returns True or False

    row_vals = sudoku[row]
    if guess in row_vals:
        return False 

   
    col_vals = []
    for i in range(9):
        col_vals.append(sudoku[i][col])

    
    row_start = (row // 3) * 3 
    col_start = (col // 3) * 3

    for r in range(row_start, row_start + 3):
        for c in range(col_start, col_start + 3):
            if sudoku[r][c] == guess:
                return False

    return True

def solve_sudoku(sudoku):
    row, col = find_empty_space(sudoku)
    
    if row is None:
        return True 
    
    
    for guess in range(1, 10): 
        if validity(sudoku, guess, row, col):
            sudoku[row][col] = guess
            if solve_sudoku(sudoku): #recursive call for solver
                return True
        
        sudoku[row][col] = -1

    return False #no possible solution

