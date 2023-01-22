import time
from functools import wraps
from typing import Any, Callable


def stop_watch(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kargs: Any) -> Any:
        print(f"{func.__name__}")
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"実行時間:{elapsed_time:.3e}[s]")
        return result

    return wrapper
