from typing import Any, List


def select_options(options: List[Any]):
    while True:
        for o, option in enumerate(options):
            print(f"{o+1}. {option}")
        res = input("Input: ")

        try:
            res = int(res)
            if res < 1 or res > len(options):
                print(f"The allowed value is from {1} to {len(options)}.")
            else:
                return options[res-1]
        except ValueError:
            print("The input should be an integer.")
            