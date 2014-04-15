
DEBUG = True

def debug(*args):
    if DEBUG:
        print(*args)
        
def setDebug(state):
    global DEBUG
    DEBUG = state