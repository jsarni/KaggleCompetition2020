from random import randint

def getRandomModelID():
    uid = randint(0, 10000000)
    return "{:07d}".format(uid)
