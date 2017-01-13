
#DEBUG = True
DEBUG = False  # Turn DEBUG off to run tests

def debug(*args):
    if DEBUG:
        print(*args)
        
def setDebug(state):
    global DEBUG
    DEBUG = state
