import streamlit as st 
import time
import sys
import cv.puzzleExtract as puzzleExtract
import cv.numberExtract as numberExtract
import solving.sudoku as solve

st.sidebar.markdown(
    """<h1>Sudoku Solver</h1>
    <h2> Using CV and Backtracking</h2>
    <p>
    Upload an image of a sudoku grid of the below format and get the solved state.
    </p>
    <img src='https://raw.githubusercontent.com/CVxTz/sudoku_solver/master/solver/samples/wiki_sudoku.png'
         width="300"></br>
    """,
    unsafe_allow_html=True,
)

file = st.file_uploader("Upload Sudoku Image", type=["jpg", "png"])


if file:
   image = puzzleExtract.read_from_file(file)
   image = puzzleExtract.readPuzzle(image)

   grid = numberExtract.extract(image)
   print(grid)

   startTime = time.time()
   puzzleSolution = solve.solve_sudoku(grid)
   timeTaken = time.time() - startTime  


         

   print(puzzleSolution)