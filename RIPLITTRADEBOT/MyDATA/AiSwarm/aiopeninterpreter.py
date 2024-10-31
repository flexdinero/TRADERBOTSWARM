'''
1. AI to watch youtube videos and read papers to understand the strategy
2. the 2nd ai will then code a backtest of those strategies
(wip) 3. third ai will execute the code and then debug it if it fails
4. production ready backtest...
___
i can build the bot for now... i just want to finish
'''

import traceback
import re
from unittest.mock import patch
from interpreter import interpreter
import dontshareconfig as d
import os

os.environ['OPENAI_API_KEY'] = d.key

def run_backtest():
    try:
        from aibacktest_script import run_faulty_backtest
        run_faulty_backtest()
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Error occurred: {error_message}")
        diagnose_and_fix_loop(error_message)

def diagnose_and_fix_loop(error_message):
    while True:
        suggestion = get_suggestion_from_interpreter(error_message)
        print(f"Suggested fix: {suggestion}")

        fixed_code = extract_code(suggestion)
        if fixed_code:
            apply_fix_with_interpreter(fixed_code)
            print("Applied the suggested fix.")
            try:
                from aibacktest_script import run_faulty_backtest
                run_faulty_backtest()
                print("No more errors. Backtest ran successfully.")
                break
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"Error occurred: {error_message}")
        else:
            print("No code fix found in the suggestion.")
            break

def get_suggestion_from_interpreter(error_message):
    interpreter.llm.model = "gpt-4.0"
    interpreter.llm.max_tokens = 30000
    interpreter.offline = True

    suggestion_prompt = f"""
    An error occurred in the following backtesting script:
    ```
    {error_message}
    ```
    Please analyze the error and suggest possible fixes to resolve it. Ensure all necessary imports are included.
    """

    with patch('builtins.input', return_value='y'):
        return interpreter.chat(suggestion_prompt)

def extract_code(suggestion):
    if isinstance(suggestion, list):
        suggestion = ' '.join([item.get('content', '') if isinstance(item, dict) else str(item) for item in suggestion])

    code_snippet = re.search(r'```python\n(.*?)\n```', suggestion, re.DOTALL)
    if code_snippet:
        return code_snippet.group(1)
    return None

def apply_fix_with_interpreter(fixed_code):
    # Command to apply the fix directly to the backtest_script.py file
    apply_prompt = f"""
    The following code needs to be applied to the file `backtest_script.py`:
    ```python
    {fixed_code}
    ```
    Please ensure the necessary imports are included and apply this code correctly to fix the issues.
    """
    with patch('builtins.input', return_value='y'):
        interpreter.chat(apply_prompt)

if __name__=="_main_":
    run_backtest()