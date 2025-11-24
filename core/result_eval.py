# core/result_eval.py

def tool_success(tool_name: str, result) -> bool:
    """
    Evaluates a tool's raw output to determine if it was a success.
    """
    # Fail if the result is a dictionary with an "error" key
    if isinstance(result, dict) and "error" in result:
        return False

    # Fail on string results that start with [ERROR]
    if isinstance(result, str) and result.strip().startswith("[ERROR]"):
        return False

    return True