import os
import os.path


def abspath(path: str, wd: str = None) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    
    if wd is None:
        wd = os.getcwd()
    elif not os.path.isabs(wd):
        wd = os.path.abspath(wd)
    
    return os.path.normpath(os.path.join(wd, path))
