from screenshot import read_board, click_board
import pyautogui
import random
from environment import Board, get_best_move, get_best_move_average_value
from agents import Agent, ValueAgent

import time


def take_move(action: int):
    # move the mouse to the location
    click_board(action)


def play_turn():
    player = 0
    parsed_board, rolled_num, game_over, winner = read_board()
    if game_over:
        return game_over, winner

    board = Board()

    board.board = parsed_board

    actions = board.get_valid_moves(player)
    if len(actions) == 0:
        raise Exception
    print("Thinking....")
    action = get_best_move_average_value(board, player, rolled_num)
    # action = actions[random.randint(0,len(actions)-1)]
    parsed_board, rolled_num, game_over, winner = read_board()
    # make sure the game isn't over
    if not game_over:
        take_move(action)
    return game_over, winner


value_agent = ValueAgent()
value_agent.load("value_agent.pt")


def play_turn_agent():
    player = 0
    parsed_board, rolled_num, game_over, winner = read_board()
    if game_over:
        return game_over, winner

    board = Board()

    board.board = parsed_board

    actions = board.get_valid_moves(player)
    if len(actions) == 0:
        raise Exception
    print("Thinking....")
    action = value_agent.get_action(player, parsed_board, rolled_num)
    # action = actions[random.randint(0,len(actions)-1)]
    parsed_board, rolled_num, game_over, winner = read_board()
    # make sure the game isn't over
    if not game_over:
        take_move(action)
    return game_over, winner


def nuclear():
    pyautogui.press("e")
    pyautogui.press("d")
    pyautogui.press("e")


def start_game():
    print("Select Opponent")
    # press e on the table
    pyautogui.press("e")
    # left 3
    [pyautogui.press("a") for i in range(3)]
    # select person
    pyautogui.press("e")
    # press right 4
    [pyautogui.press("d") for i in range(4)]
    # wait
    time.sleep(1)
    # e to start
    pyautogui.press("e")
    # wait
    time.sleep(1)
    # play
    print("Start game")
    for i in range(50):
        try:
            game_over, winner = play_turn_agent()
        except Exception as e:
            print(f"we got exception {e}")
            nuclear()
            return
        if game_over:
            print(f"We have a winner {winner}")
            # leave game
            pyautogui.press("e")
            time.sleep(5)
            return
        time.sleep(7)


while True:
    start_game()
