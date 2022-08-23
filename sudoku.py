def find_zero(arr,pos):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]==0):
                pos[0] = row
                pos[1] = col
                return True
    return False

def row_val(arr, row, value):
    for i in range(9):
        if(arr[row][i]==value):
            return True
    return False

def col_val(arr,col,value):
    for i in range(9):
        if(arr[col][i]):
            return True
    return False

def grid_val(arr,row,col,value):
    for i in range(3):
        for j in range(3):
            if(arr[i+row][j+col]==value):
                return True
    return False

def validity(arr,row,col,value):
    return row_val(arr,row,value) and col_val(arr,col,value) and grid_val(arr,row-row%3,col-col%3,value)

def solve(arr):
    pos=[0,0]
    
    if(not find_zero(arr,pos)):
        return True
    
    row = pos[0]
    col = pos[1]
    
    for value in range(1,10):
        if(validity(arr,row,col,value)):
            arr[row][col]=value
            if(solve(arr)):
                return True
            arr[row][col] = 0
            
    return False

def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j])
            print('\n')
            
def final_sudoku(grid):
    if(solve(grid)):
        print('---')
    else:
        print("NO POSSIBLE SOLUTION FOR GIVEN SUDOKU")
        grid = grid.astype(int)
        return grid