# General
import time
import random
import string
import numpy as np
from collections import deque
import copy
import os
import glob
import sys

# Connection
from telnetlib import Telnet

# Pathfinding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder

# Global variables
nMaps = 20                          # number of maps to save
listOfMaps = deque(maxlen=nMaps)    # a list of the latest maps
tempMap = deque(maxlen=1)
host = 'margot.di.unipi.it'
serverPort = 8421
chatPort = 8422
train_timer = 0.005
timer = 0.5
ai_timer = 0.3

# version
#{group-number}-{ai-version}
ai_version = '0.52'

class CommunicationHandler:

  def __init__(self, host, port):
    self.host = host
    self.port = port
    self.tn = self.connect(host, port)

    self.lastAction = time.time()
    self.timer = timer  # how many seconds do I have to wait before performing another action?


  def connect(self, host, port):
    tn = Telnet(host, port)
    print('Connected!')
    return tn


  def createGame(self, matchName, params=''):

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Creating the new game
    command = 'NEW ' + matchName + ' ' + params + '\n'
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[CREATE] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res


  def joinGame(self, matchName, playerName, train=False, observer=False):
    self.matchName = matchName
    self.playerName = playerName
    command = matchName + ' JOIN ' + playerName + ' AI _ AI0-' + ai_version + '\n'
    if (observer):
      command = matchName + ' JOIN ' + playerName + ' O -\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    if (train or observer):
      self.timer = train_timer
    else:
      self.timer = ai_timer

    # Joining game
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[JOIN] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res


  def startGame(self):
    command = self.matchName + ' START\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Starting game
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[START] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res
    

  def getMap(self):
    command = self.matchName + ' LOOK' + '\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Getting the map
    self.tn.write(command.encode('utf8'))
    message = self.tn.read_until('»\n'.encode('utf8')).decode('utf8')
    if ('ERROR' in message):
      return -1
    res = message.split('OK LONG\n')

    # Update last action performed's timestamp
    self.lastAction = time.time()

    tempMap.append(res[1])
    return toMatrix(res[1])


  def getStatus(self):
    command = self.matchName + ' STATUS' + '\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Getting the status
    self.tn.write(command.encode('utf8'))
    message = self.tn.read_until('»\n'.encode('utf8')).decode('utf8')
    if ('ERROR' in message):
      return -1
    res = message.split('OK LONG\n')

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res[1].split('«ENDOFSTATUS»')[0]


  def move(self, direction):
    command = self.matchName + ' MOVE ' + direction + '\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Moving
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[MOVE-' + direction + '] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res


  def shoot(self, direction):
    command = self.matchName + ' SHOOT ' + direction + '\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Shooting
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split('\n')[0].split(' ')
    print('[SHOOT-' + direction + '] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res
    
  def sendNOP(self):
    command = self.matchName + ' NOP\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Send NOP
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[NOP] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res

  def accuse(self, playerName):
    command = self.matchName + ' ACCUSE ' + playerName + '\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Accusing
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[ACCUSE-' + playerName + '] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res

  def judge(self, playerName, nature):
    command = self.matchName + ' JUDGE ' + playerName + ' ' + nature + '\n'

    # sleep until I can send another command
    currentTime = time.time()
    sleepingTime = self.timer - (currentTime - self.lastAction)
    if (sleepingTime<0):
      sleepingTime=0
    time.sleep(sleepingTime)

    # Judge
    self.tn.write(command.encode('utf8'))
    res = self.tn.read_until('\n'.encode('utf8')).decode('utf8').split(' ')
    print('[JUDGE-' + playerName + '] ' + res[0])

    # Update last action performed's timestamp
    self.lastAction = time.time()
    return res


# phrase list
CHAT_DICTIONARY = {
  'salute': ['hi', 'hello', 'hi there', 'hello guys', 'good evening', 'glhf', 'hello there', 'don\'t feed please'],
  'victory' : ['too easy', 'nice game!', 'ggez', 'ez', 'another one?'],
  'defeat' : ['whatever', 'ggwp', 'report my team please', 'wtf', 'hacker!!', 'LAAAAAG', 'great job', 'another one?'],
  'attack' : ['let\'s attack!', 'let\'s go forward', 'attack them!', 'let\'s go guys!'],
  'defend' : ['come at me!', 'i\'ll defend the flag', 'defending flag', 'defending', 'def here', 'going to base'],
  'kill' : ['rush B', 'burn \'em all!', 'gonna kill you all'],
  'hide' : ['let\'s wait here', 'guys, play safe', 'waiting for them']
}

class Chat:
  def __init__(self, host, port, player):
    self.host = host
    self.port = port
    self.tn = self.connect(host, port, player)

  def connect(self, host, port, player):
    tn = Telnet(host, port)

    # Connecting to the game chat
    command = 'NAME ' + player.name + '\n'
    tn.write(command.encode('utf8'))

    print('Connected!')
    return tn

  def join(self, channel, team):
    self.team = team
    if (team == 1):
      self.teamChannelName = channel + ':1'
      self.enemyChannelName = channel + ':0'
    else:
      self.teamChannelName = channel + ':0'
      self.enemyChannelName = channel + ':1'

    # Joining channels
    self.tn.write(('JOIN '+channel+'\n').encode('utf8'))
    self.tn.write(('JOIN '+self.teamChannelName+'\n').encode('utf8'))
    self.tn.write(('JOIN '+self.enemyChannelName+'\n').encode('utf8'))
    
    
    return True

  def leave(self, channel):
    self.tn.write(('LEAVE '+channel).encode('utf8'))
    
    return True

  def post(self, channel, text):
    command = 'POST ' + channel + ' ' + text + '\n'
    self.tn.write(command.encode('utf8'))

    return True

  def readChat(self, timeout=None):
    res = self.tn.read_until('\n'.encode('utf8'), timeout=timeout).decode('utf8')

    return res


class Player:

  def __init__(self, symbol, name, team, posX=-1, posY=-1):
    self.symbol = symbol
    self.name = name
    self.team = team
    self.loyalty = self.team #for the other players
    self.energy = -1
    self.score = -1
    self.posX = posX
    self.posY = posY
    self.alive = True 
    self.posHistory = []
    self.posHistory.append((posY,posX))
    if (symbol.islower()):
      self.myFlag = 'x'
      self.enemyFlag = 'X'
    else:
      self.myFlag = 'X'
      self.enemyFlag = 'x'


  def addInfo(self, energy, score, loyalty):
    self.energy = energy
    self.score = score
    self.loyalty = loyalty
    if (self.loyalty != self.team):
      self.myFlag, self.enemyFlag = self.enemyFlag, self.myFlag

  def updateLoyalty(self, val):
    if (self.loyalty != val):
      self.loyalty = val
      self.myFlag, self.enemyFlag = self.enemyFlag, self.myFlag

  def lastPosition(self, posX, posY):
    self.posX = int(posX)
    self.posY = int(posY)
    self.posHistory.append((posY, posX))

  def setFlagDistance(self, dist):
    self.distFromFlag = dist

  def printInfo(self):
    if (self.score == -1):
      print('sym=' + self.symbol + ', team='+ self.team)
    else:
      print('sym=' + self.symbol + ', team='+ self.team + ', loy='+ self.loyalty + ', ene='+ str(self.energy) + ', sco='+ str(self.score))

# Extracts informations about the player from the getStatus command
def extractMyInfo(status):
  playerInfo = status.split('ME: ')[1]
  symbol = playerInfo.split('symbol=')[1][0]
  name = playerInfo.split('name=')[1].split(' ')[0]
  team = int(playerInfo.split('team=')[1][0])
  loyalty = int(playerInfo.split('loyalty=')[1][0])
  energy = int(playerInfo.split('energy=')[1].split(' ')[0])
  score = int(playerInfo.split('score=')[1].split('\n')[0])

  stat = status.split('PL: ')[1:]
  for line in stat:
    playerSymbol = line.split('symbol=')[1][0]
    if (symbol == playerSymbol):
      posX = int(line.split('x=')[1].split(' ')[0])
      posY = int(line.split('y=')[1].split(' ')[0])
      break
  
  myPlayer = Player(symbol, name, team, posX, posY)
  myPlayer.addInfo(energy, score, loyalty)

  return myPlayer

# Returns the state of the game (LOBBY, FINISHED, ...)
def gameState(status):
  return status.split('state=')[1].split(' ')[0]

# Extracts info about ANY player
def extractPlayerInfo(status):
  playersStatus = status.split('PL: ')[1:]
  players = {}
  for line in playersStatus:
    symbol = line.split('symbol=')[1][0]
    name = line.split('name=')[1].split(' ')[0]
    team = int(line.split('team=')[1][0])
    posX = int(line.split('x=')[1].split(' ')[0])
    posY = int(line.split('y=')[1].split(' ')[0])
    state = line.split('state=')[1].split('/n')[0]
    if (symbol != myPlayer.symbol):
      players[symbol] = Player(symbol, name, team, posX, posY)
      if ('ACTIVE' not in state and 'LOBBYGUEST' not in state):
        players[symbol].alive = False
    else:
      myPlayer.lastPosition(posX, posY)

  return players

def updateMapFromStatus(map, myPlayer, players):
  for player in players.values():
    p_posX, p_posY = player.posX, player.posY
    oldY, oldX = findOnMap(map, player.symbol)
    map[oldY][oldX] = '.' # putting some grass
    map[p_posY][p_posX] = player.symbol
  
  p_posX, p_posY = myPlayer.posX, myPlayer.posY
  oldY, oldX = findOnMap(map, myPlayer.symbol)
  map[oldY][oldX] = '.' # putting some grass
  map[p_posY][p_posX] = myPlayer.symbol

  return map


def updatePlayerPosition(status, players):
  playersStatus = status.split('PL: ')[1:]
  for line in playersStatus:
    symbol = line.split('symbol=')[1][0]
    posX = line.split('x=')[1].split(' ')[0]
    posY = line.split('y=')[1].split(' ')[0]
    state = line.split('state=')[1].split('/n')[0]
    if (symbol != myPlayer.symbol):
      players[symbol].lastPosition(posX, posY)
      if ('ACTIVE' not in state and 'LOBBYGUEST' not in state):
        players[symbol].alive = False
    else:
      myPlayer.lastPosition(posX, posY)
      if ('ACTIVE' not in state and 'LOBBYGUEST' not in state):
        myPlayer.alive = False

  return players


# Terrains in the game
"""
. grass, freely walkable, allow shooting
# wall, not walkable, stops shoots
~ river, walkable, cannot shoot while on it, allow shooting through it
@ ocean, not walkable, allow shooting through it
! trap, will subtract energy from player if walked on, allow shooting through it
"""

# values for map handling
flag_value = 2
factor = 8

# Converts map from string to char matrix
def toMatrix(map):
  matrix = list(map.split('\n')[:-2])
  newMap = []
  for row in matrix:
    newMap.append(list(row))
  listOfMaps.append(newMap)
  return newMap

# Converts symbols into numbers for A*
# <=0 obstacles
# >0  cost to pass through
def prepareMap(map, player):
  for row in range(len(map)):
    for col in range(len(map[row])):
      if (map[row][col] == '#'): #wall
        map[row][col] = -1
      elif (map[row][col] == '.'): #grass
        map[row][col] = 1
      elif (map[row][col] == '~'): #river
        map[row][col] = 1.1
      elif (map[row][col] == '@'): #ocean
        map[row][col] = -3
      elif (map[row][col] == '!'): #trap
        map[row][col] = 5
      elif (map[row][col] == '$'): #energy recharge
        map[row][col] = 1
      elif (map[row][col] == '&'): #barrier
        map[row][col] = -2
      elif (map[row][col] == player.symbol): #myself
        map[row][col] = 1
      elif (map[row][col] == player.myFlag): #my flag
        map[row][col] = -5
      elif (map[row][col] == player.enemyFlag): #enemy Flag
        map[row][col] = 1
        if (player.loyalty != player.team): #the player is an impostor, can't walk on flags
          map[row][col] = -6
      else: #other players or enemy flags
        map[row][col] = 1

  return map


# Converts symbols into numbers for Q-Learning
# 0 wall
# 1 grass
# 2 river
# 3 ocean
# 4 traps
# 5 boosts
# 6 barrier
# 7 flag X
# 8 flag x
# 9 barrier
# 10 teamflag
# 11 opponentflag
def prepareMap2(original_map):
  map = np.zeros((128,256)) # setting matrix to maximum size
  map = np.zeros((len(original_map),len(original_map[0])))
  
  for row in range(len(original_map)):
    for col in range(len(original_map[row])):
      if (original_map[row][col] == '#'): #wall
        map[row][col] = 0
      elif (original_map[row][col] == '.'): #grass
        map[row][col] = 1
      elif (original_map[row][col] == '~'): #river
        map[row][col] = 2
      elif (original_map[row][col] == '@'): #ocean
        map[row][col] = 3
      elif (original_map[row][col] == '!'): #trap
        map[row][col] = 4
      elif (original_map[row][col] == '$'): #energy recharge
        map[row][col] = 5
      elif (original_map[row][col] == '&'): #barrier
        map[row][col] = 6
      elif (original_map[row][col] == 'X'): #flag 1
        map[row][col] = 7
      elif (original_map[row][col] == 'x'): #flag 2
        map[row][col] = 8
      elif (original_map[row][col].isupper()): #uppercase team
        map[row][col] = ord(original_map[row][col])-56
      else:
        map[row][col] = ord(original_map[row][col])-68
 
  return map


def prepareMap3(original_map, myPlayer, players, size=(64,128)):
  increase = 30
  min_increase = 10
  ocean = -3
  wall = -1
  barrier = -2

  # decrease increase value as the distance grows
  map = prepareMap(original_map, myPlayer)
  for player in players.values():
    # check columns and rows of the enemies and set them to higher values
    if (player.team != myPlayer.team and player.alive): 
      # from posX to the right ->
      count = 0
      for column in range(player.posX, size[1]):
        if (map[player.posY][column] == wall or map[player.posY][column] == barrier
            or (original_map[player.posY][column] in players.keys() and  not players[original_map[player.posY][column]].alive)):
          break
        elif (map[player.posY][column] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[player.posY][column] += max_val
          count += 1
      # from posX to the left <-
      count = 0
      for column in range(player.posX, -1, -1):
        if (map[player.posY][column] == wall or map[player.posY][column] == barrier
             or (original_map[player.posY][column] in players.keys() and  not players[original_map[player.posY][column]].alive)):
          break
        elif (map[player.posY][column] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[player.posY][column] += max_val
          count += 1
      # from posY to the top /|\
      for row in range(player.posY, -1, -1):
        if (map[row][player.posX] == wall or map[row][player.posX] == barrier
             or (original_map[row][player.posX] in players.keys() and  not players[original_map[row][player.posX]].alive)):
          break
        elif (map[row][player.posX] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[row][player.posX] += max_val
          count += 1
      # from posY to the bottom \|/
      for row in range(player.posY, size[0]):
        if (map[row][player.posX] == wall or map[row][player.posX] == barrier
            or (original_map[row][player.posX] in players.keys() and  not players[original_map[row][player.posX]].alive)):
          break
        elif (map[row][player.posX] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[row][player.posX] += max_val
          count += 1
      # set the enemy position to 11
      map[player.posY][player.posX] = increase+1
    
  return map

def prepareImpostorMap(original_map, myPlayer, players, size=(64,128)):
  increase = 30
  min_increase = 10
  ocean = -3
  wall = -1
  barrier = -2

  # decrease increase value as the distance grows
  map = prepareMap(original_map, myPlayer)
  for player in players.values():
    # check columns and rows of the enemies and set them to higher values
    if (player.alive): 
      # from posX to the right ->
      count = 0
      for column in range(player.posX, size[1]):
        if (map[player.posY][column] == wall or map[player.posY][column] == barrier
            or (original_map[player.posY][column] in players.keys() and  not players[original_map[player.posY][column]].alive)):
          break
        elif (map[player.posY][column] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[player.posY][column] += max_val
          count += 1
      # from posX to the left <-
      count = 0
      for column in range(player.posX, -1, -1):
        if (map[player.posY][column] == wall or map[player.posY][column] == barrier
            or (original_map[player.posY][column] in players.keys() and  not players[original_map[player.posY][column]].alive)):
          break
        elif (map[player.posY][column] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[player.posY][column] += max_val
          count += 1
      # from posY to the top /|\
      for row in range(player.posY, -1, -1):
        if (map[row][player.posX] == wall or map[row][player.posX] == barrier
            or (original_map[row][player.posX] in players.keys() and  not players[original_map[row][player.posX]].alive)):
          break
        elif (map[row][player.posX] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[row][player.posX] += max_val
          count += 1
      # from posY to the bottom \|/
      for row in range(player.posY, size[0]):
        if (map[row][player.posX] == wall or map[row][player.posX] == barrier
            or (original_map[row][player.posX] in players.keys() and  not players[original_map[row][player.posX]].alive)):
          break
        elif (map[row][player.posX] == ocean):
          count += 1
          continue
        else:
          max_val = max(min_increase, increase-count*1.5)
          map[row][player.posX] += max_val
          count += 1
      # set the enemy position to 11
      map[player.posY][player.posX] = increase+1
    
  return map
    
def findZone(posX, posY):
  return int(posX/factor),int(posY/factor)

def findAdjacientZones(zones, posX, posY):
  myZoneX, myZoneY = findZone(posX, posY)
  zone_up, zone_down, zone_left, zone_right = (-1,-1), (-1,-1), (-1,-1), (-1,-1)
  if (myZoneY > 0):
    zone_up = (myZoneX, myZoneY-1)
  if (myZoneY < zones.shape[0]-1):
    zone_down = (myZoneX, myZoneY+1)
  if (myZoneX > 0):
    zone_left = (myZoneX-1, myZoneY)
  if (myZoneX < zones.shape[1]-1):
    zone_right = (myZoneX+1, myZoneY)

  return zone_up, zone_down, zone_left, zone_right
 
'''
  returns:
    - best =  best value among the adjacient zones
    - bestX, bestY =  best cell in the macro_area (real coordinates)
'''
def bestAdjacientZone(zones, best_pos, posX, posY):
  zone_up, zone_down, zone_left, zone_right = findAdjacientZones(zones, posX, posY)
  best_up, best_down, best_left, best_right = 9999, 9999, 9999, 9999
  if (zone_up != (-1,-1)):
    best_up = zones[zone_up[1]][zone_up[0]]
  if (zone_down != (-1,-1)):
    best_down = zones[zone_down[1]][zone_down[0]]
  if (zone_left != (-1,-1)):
    best_left = zones[zone_left[1]][zone_left[0]]
  if (zone_right != (-1,-1)):
    best_right = zones[zone_right[1]][zone_right[0]]
  
  best = best_up
  (bestX,bestY) = best_pos[zone_up[1]][zone_up[0]]
  if (best > best_down):
    best = best_down
    (bestX,bestY) = best_pos[zone_down[1]][zone_down[0]]
  if (best > best_left):
    best = best_left
    (bestX,bestY) = best_pos[zone_left[1]][zone_left[0]]
  if (best > best_right):
    best = best_right
    (bestX,bestY) = best_pos[zone_right[1]][zone_right[0]]

  return best, bestX, bestY


def computeZone(map, row, col, flagX, flagY):
  sum = 0
  counter = 0
  posX, posY = -1, -1
  val = 9999
  flag_dist = 9999 
  for i in range(row, row+factor):
    for j in range(col, col+factor):
      if (map[i][j] > 0):
        if (map[i][j] < val):
          posX, posY = j, i
          val = map[i][j]
          flag_dist = distance(posX, posY, flagX, flagY)
        elif (map[i][j] == val):
          dist_old = distance(posX, posY, flagX, flagY)
          dist_new = distance(j, i, flagX, flagY)
          if (dist_new < dist_old):
            posX, posY = j, i
            flag_dist = distance(posX, posY, flagX, flagY)
        counter += 1
        sum += map[i][j]
        
  if (counter == 0):
    counter = 1

  return sum/counter + flag_dist, (posX,posY)


def computeDangerMap(map, enemy_flagX, enemy_flagY):
  n_row = len(map)
  n_col = len(map[0])

  zones = np.zeros((int(n_row/factor),int(n_col/factor)))
  best_pos = np.zeros((int(n_row/factor),int(n_col/factor)), dtype='i,i')

  flag_zone_row, flag_zone_col = findZone(enemy_flagX, enemy_flagY)

  for row in range(zones.shape[0]):
    for col in range(zones.shape[1]):
      avg, pos = computeZone(map, row*factor, col*factor, enemy_flagX, enemy_flagY)
      zones[row][col], best_pos[row][col] = avg, pos

      if(row == flag_zone_row and col == flag_zone_col):
        zones[row][col] -= flag_value

  return zones, best_pos

def computeImpostorZone(map, row, col):
  sum = 0
  counter = 0
  posX, posY = -1, -1
  val = 9999
  for i in range(row, row+factor):
    for j in range(col, col+factor):
      if (map[i][j] > 0):
        if (map[i][j] < val):
          posX, posY = j, i
          val = map[i][j]
        counter += 1
        sum += map[i][j]

  return sum/counter, (posX,posY)

def computeImpostorDangerMap(map, players):
  n_row = len(map)
  n_col = len(map[0])

  zones = np.zeros((int(n_row/factor),int(n_col/factor)))
  best_pos = np.zeros((int(n_row/factor),int(n_col/factor)), dtype='i,i')

  for row in range(zones.shape[0]):
    for col in range(zones.shape[1]):
      avg, pos = computeImpostorZone(map, row*factor, col*factor)
      zones[row][col], best_pos[row][col] = avg, pos

  return zones, best_pos

# Finds an item on the map
def findOnMap(map, symbol):
  for row in range(len(map)):
    for col in range(len(map[row])):
      if (map[row][col] == symbol):
        return row, col
  return -1, -1


colors = {
    "team_1":       [220,20,60],    #team_1       = red
    "team_1_flag":  [139,0,0],      #team_1_flag  = dark_red
    "team_2":       [65,105,225],   #team_2       = blue
    "team_2_flag":  [0,0,153],      #team_2_flag  = dark_blue
    "grass":        [153,255,153],  #grass        = green
    "ocean":        [0,191,255],    #ocean        = light_blue
    "river":        [204,255,255],  #river        = lightest_blue
    "wall":         [0,0,0],        #wall         = black
    "barrier":      [192,192,192],  #barrier      = grey
    "energy":       [255,255,255],  #energy       = white
    "trap":         [255,102,255]   #trap         = purple
}
def buildMap(map):
  beautiful_map = np.zeros((len(map), len(map[0]), 3), dtype=np.uint8)
  for row in range(len(map)):
    for col in range(len(map[row])):
      if (map[row][col] == '#'): #wall
        beautiful_map[row][col] = colors["wall"]
      elif (map[row][col] == '.'): #grass
        beautiful_map[row][col] = colors["grass"]
      elif (map[row][col] == '~'): #river
        beautiful_map[row][col] = colors["river"]
      elif (map[row][col] == '@'): #ocean
        beautiful_map[row][col] = colors["ocean"]
      elif (map[row][col] == '!'): #trap
        beautiful_map[row][col] = colors["trap"]
      elif (map[row][col] == '$'): #energy recharge
        beautiful_map[row][col] = colors["energy"]
      elif (map[row][col] == '&'): #barrier
        beautiful_map[row][col] = colors["barrier"]
      elif (map[row][col] == 'X'): #flag 1
        beautiful_map[row][col] = colors["team_1_flag"]
      elif (map[row][col] == 'x'): #flag 2
        beautiful_map[row][col] = colors["team_2_flag"]
      elif (map[row][col].isupper()): #uppercase team
        beautiful_map[row][col] = colors["team_1"]
      else:
        beautiful_map[row][col] = colors["team_2"]

  return beautiful_map

def printMap(map, size=(10,10)):
  beautiful_map = buildMap(map)
  plt.figure(figsize=size)
  plt.imshow(beautiful_map, interpolation='nearest')
  plt.xticks([])
  plt.yticks([])
  plt.show()

# https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python/44951066#44951066
def createReplay(map_h, filename):
  os.system("mkdir frames")

  i=0
  for map in map_h:
    # frame creation
    beautiful_map = buildMap(map)
    plt.imshow(beautiful_map, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    # save the frame in memory
    plt.savefig('frames/img' + '{:04d}'.format(i) + '.png')
    i = i+1

  # create videos from frames (3fps, final video:filename.mp4)
  os.system("ffmpeg -r 3 -i frames/img%04d.png -vcodec mpeg4 -y " + filename + ".mp4")

  # removing all frames
  files = glob.glob('frames/*.png')
  for f in files:
    open(f, 'w').close() #overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
    os.remove(f) #delete the blank file from google drive will move the file to bin instead


# Manhattan distance
def distance(xa, ya, xb, yb):
  return (abs(xa - xb) + abs(ya - yb))
  
# Returns a path, calculated using A*
def findPath(map, startx, starty, endx, endy):
  grid = Grid(matrix=map)
  # (x, y) = (col, row)
  start = grid.node(startx, starty)
  end = grid.node(endx, endy)

  finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
  path, runs = finder.find_path(start, end, grid)
  return path

# Returns a matrix of distances from the flag
def findDistances(map, endx, endy):
  distMap = copy.deepcopy(map)
  for row in range(len(map)):
    for col in range(len(map[row])):
      # if the point is valid (value > 0)
      if(map[row][col]>0):
        path = findPath(map, col, row, endx, endy)
        distMap[row][col] = len(path)
      # if the point is not valid
      else:
        distMap[row][col] = -1
  return distMap

# Takes current position and next position, finds the direction
def findNextDirection(posX, posY, nextX, nextY):
  if (nextX < posX):
    return 'W'
  elif (nextX > posX):
    return 'E'
  elif (nextY < posY):
    return 'N'
  elif (nextY > posY):
    return 'S'
  return ''

# follows a path from start to end
def pathToCommand(ch, path):
  if (len(path) == 0):
    return -1
  posX, posY = path[0]
  for i in range(1,len(path)):
    nextX, nextY = path[i]
    result = findNextDirection(posX, posY, nextX, nextY)
    if(result != ''):
      ch.move(result)
    posX, posY = nextX, nextY
  print('[MOVE] Done!')
  return 1

def printMatrix(matrix):
  for row in matrix:
    print(row)


def movedTowardsFlag(map, player, direction):
  enemy_flag_x, enemy_flag_y = findOnMap(map, player.enemyFlag)
  previous_dist = distance(player.posX, player.posY, enemy_flag_x, enemy_flag_y)
  if (direction == "N"):
    new_dist = distance(player.posX, player.posY-1, enemy_flag_x, enemy_flag_y)
  elif (direction == "S"):
    new_dist = distance(player.posX, player.posY+1, enemy_flag_x, enemy_flag_y)
  elif (direction == "W"):
    new_dist = distance(player.posX-1, player.posY, enemy_flag_x, enemy_flag_y)
  elif (direction == "E"):
    new_dist = distance(player.posX+1, player.posY, enemy_flag_x, enemy_flag_y)
  return new_dist < previous_dist

def defendPosition(dumb_ch, map, players, myPlayer, iter=5):
  print('[DEF] Defending Position.')
  # Rambo mode 
  shoot = False
  i = 0 # number of actions
  start_timer = time.time()
  while (not shoot and i < iter):
    if (myPlayer.energy == 0): 
      break

    old_positions = {}
    for p in players.values():
      old_positions[p.symbol] = (p.posX, p.posY)
    status = dumb_ch.getStatus()
    myPlayer = extractMyInfo(status)
    players = extractPlayerInfo(status)
    players = updatePlayerPosition(status, players)
    map = updateMapFromStatus(map, myPlayer, players)
    lap_timer = time.time() - start_timer 
    print('[DEF] Defending Position iter:' + str(i) + ' - timer: ' + str(lap_timer))
    new_positions = {}
    for p in players.values():
      new_positions[p.symbol] = (p.posX, p.posY)
    # how much should i wait in this position?  
    min_x, min_y, min_player_x, min_player_y = calculateClosest(players, myPlayer)
    # impostor case: find the best position and wait
    if (myPlayer.team != myPlayer.loyalty):
      if (min_player_x == '*' or min_player_y == '*'):
        break
    # no enemies found, empty list -> run to the enemy flag
    if (min_player_x == '*' or min_player_y == '*' and myPlayer.loyalty == myPlayer.team):
      myPreparedMap = prepareMap3(copy.deepcopy(map), myPlayer, players, size=(len(map),len(map[0])))
      enemyFlagY, enemyFlagX = findOnMap(map, myPlayer.enemyFlag)
      myPath = findPath(myPreparedMap, myPlayer.posX, myPlayer.posY, enemyFlagX, enemyFlagY)
      if (len(myPath) > 0):
        p = pathToCommand(dumb_ch, myPath) # run to the flag
        break

    closer_player_x = players[min_player_x]
    closer_player_y = players[min_player_y]
    # check if the closest player is in shooting range
    if (closer_player_x.posX - myPlayer.posX == 1): #enemy from E
      shootDirection = findNextDirection(myPlayer.posX, myPlayer.posY, closer_player_x.posX-1, closer_player_x.posY)
      shoot_dist = checkShoot(map, myPlayer, closer_player_x.symbol, shootDirection)
      if (shoot_dist > abs(myPlayer.posY - closer_player_x.posY)):
        shootRes = dumb_ch.shoot(shootDirection)
        print(shootRes)
        if (shootRes[1].isalpha()):
          players[shootRes[1]].alive = False
          shoot = True
          print('[YES] Shoot hit: ' + shootRes[1])
        if (closer_player_x.alive == True): # if he is alive, shoot again 
          shootRes = dumb_ch.shoot(shootDirection)
          print(shootRes)
          if (shootRes[1].isalpha()):
            players[shootRes[1]].alive = False
            shoot = True
            print('[YES] Shoot hit: ' + shootRes[1])

    if (closer_player_x.posX - myPlayer.posX == -1): #enemy from W
      shootDirection = findNextDirection(myPlayer.posX, myPlayer.posY, closer_player_x.posX+1, closer_player_x.posY)
      shoot_dist = checkShoot(map, myPlayer, closer_player_x.symbol, shootDirection)
      if (shoot_dist > abs(myPlayer.posY - closer_player_x.posY)):
        shootRes = dumb_ch.shoot(shootDirection)
        print(shootRes)
        if (shootRes[1].isalpha()):
          players[shootRes[1]].alive = False
          shoot = True
          print('[YES] Shoot hit: ' + shootRes[1])
        if (closer_player_x.alive == True): # if he is alive, shoot again
          shootRes = dumb_ch.shoot(shootDirection)
          print(shootRes)
          if (shootRes[1].isalpha()):
            players[shootRes[1]].alive = False
            shoot = True
            print('[YES] Shoot hit: ' + shootRes[1])

    if (closer_player_y.posY - myPlayer.posY == 1): #enemy from S
      shootDirection = findNextDirection(myPlayer.posX, myPlayer.posY, closer_player_y.posX, closer_player_y.posY-1)
      shoot_dist = checkShoot(map, myPlayer, closer_player_y.symbol, shootDirection)
      if (shoot_dist > abs(myPlayer.posX - closer_player_y.posX)):
        shootRes = dumb_ch.shoot(shootDirection)
        print(shootRes)
        if (shootRes[1].isalpha()):
          players[shootRes[1]].alive = False
          shoot = True
          print('[YES] Shoot hit: ' + shootRes[1])
        if (closer_player_y.alive == True): # if he is alive, shoot again
          shootRes = dumb_ch.shoot(shootDirection)
          print(shootRes)
          if (shootRes[1].isalpha()):
            players[shootRes[1]].alive = False
            shoot = True
            print('[YES] Shoot hit: ' + shootRes[1])

    if (closer_player_y.posY - myPlayer.posY == -1): #enemy from N
      shootDirection = findNextDirection(myPlayer.posX, myPlayer.posY, closer_player_y.posX, closer_player_y.posY+1)
      shoot_dist = checkShoot(map, myPlayer, closer_player_y.symbol, shootDirection)
      if (shoot_dist > abs(myPlayer.posX - closer_player_y.posX)):
        shootRes = dumb_ch.shoot(shootDirection)
        print(shootRes)
        if (shootRes[1].isalpha()):
          players[shootRes[1]].alive = False
          shoot = True
          print('[YES] Shoot hit: ' + shootRes[1])
        if (closer_player_y.alive == True): # if he is alive, shoot again 
          shootRes = dumb_ch.shoot(shootDirection)
          print(shootRes)
          if (shootRes[1].isalpha()):
            players[shootRes[1]].alive = False
            shoot = True
            print('[YES] Shoot hit: ' + shootRes[1])

    if (closer_player_x.posX == myPlayer.posX):
      shootDirection = findNextDirection(myPlayer.posX, myPlayer.posY, closer_player_x.posX, closer_player_x.posY)
      shoot_dist = checkShoot(map, myPlayer, closer_player_x.symbol, shootDirection)
      if (shoot_dist > abs(myPlayer.posY - closer_player_x.posY)):
        shootRes = dumb_ch.shoot(shootDirection)
        print(shootRes)
        if (shootRes[1].isalpha()):
          players[shootRes[1]].alive = False
          shoot = True
          print('[YES] Shoot hit: ' + shootRes[1])
        print('[YES] Shoot!')

    if (closer_player_y.posY == myPlayer.posY):
      shootDirection = findNextDirection(myPlayer.posX, myPlayer.posY, closer_player_y.posX, closer_player_y.posY)
      shoot_dist = checkShoot(map, myPlayer, closer_player_y.symbol, shootDirection)
      if (shoot_dist > abs(myPlayer.posX - closer_player_y.posX)):
        shootRes = dumb_ch.shoot(shootDirection)
        print(shootRes)
        if (shootRes[1].isalpha()):
          players[shootRes[1]].alive = False
          shoot = True
          print('[YES] Shoot hit: ' + shootRes[1])
        print('[YES] Shoot!')
    
    i+=1

  if (not shoot):
    print('[NO] No Shoot!')


def beImpostor(player):
  return 0


def judgePlayers(judged_players, players, ch):
  for player in players.values():
    if(player.name not in judged_players):
      print("[JUDGED] " + player.name)
      ch.judge(player.name, 'AI')
      judged_players.append(player.name)
  return judged_players


def calculateSteps(players, myPlayer):
  min_x, min_y = 9999, 9999
  for player in players.values():
    if (player.team != myPlayer.team and player.alive):
      dist_x = abs(myPlayer.posX - player.posX)
      dist_y = abs(myPlayer.posY - player.posY)
      if (min_x > dist_x):
        min_x = dist_x 
      if (min_y > dist_y):
        min_y = dist_y
  steps = int(min(min_x, min_y)/2)

  if (steps > 0):
    return steps-1
  else:
    return steps

def calculateClosest(players, myPlayer):
  min_x, min_y = 9999, 9999
  min_x_player, min_y_player = '*', '*'

  for player in players.values():
    if (player.team != myPlayer.loyalty and player.alive): 
      dist_x = abs(myPlayer.posX - player.posX)
      dist_y = abs(myPlayer.posY - player.posY)
      if (min_x >= dist_x):
        min_x = dist_x
        min_x_player = player.symbol
      if (min_y >= dist_y):
        min_y = dist_y
        min_y_player = player.symbol
  return min_x, min_y, min_x_player, min_y_player


def checkShoot(map, player, target_player, direction):
    posX, posY = player.posX, player.posY
    wall = '#'
    barrier = '&'
    found = False
    distance = -1
    if (direction == 'N' and posY>0): #check up on the column
      i = posY-1
      while (not found and i>=0):
        if (wall in map[i][posX] or barrier in map[i][posX]
            or (map[i][posX] != target_player and map[i][posX].isalpha())):
          found = True
          distance = posY - i
        i-=1
    elif (direction == 'S' and posY<len(map)): #check down on the column
      i = posY+1
      while (not found and i<len(map)):
        if (wall in map[i][posX] or barrier in map[i][posX]
            or (map[i][posX] != target_player and map[i][posX].isalpha())):
          found = True
          distance = i - posY
        i+=1
    elif (direction == 'W' and posX>0): #check left on the row
      i = posX-1
      while (not found and i>=0):
        if (wall in map[posY][i] or barrier in map[posY][i]
            or (map[posY][i] != target_player and map[posY][i].isalpha())):
          found = True
          distance = posX - i
        i-=1
    elif (direction == 'E' and posX<len(map[0])): #check right on the row
      i = posX+1
      while (not found and i<len(map[0])):
        if (wall in map[posY][i] or barrier in map[posY][i]
            or (map[posY][i] != target_player and map[posY][i].isalpha())):
          found = True
          distance = i - posX
        i+=1
        
    return distance

def movingDirection(old_pos, new_pos):
  old_posX, old_posY = old_pos[0], old_pos[1]
  new_posX, new_posY = new_pos[0], new_pos[1]
  direction = ('O', 'O') # E/W, N/S
  if (old_posX < new_posX): #moved to the east
    direction[0] = 'E'
  elif (old_posX > new_posX): #moved to the west
    direction[0] = 'W'
  if (old_posY < new_posY): #moved to the south
    direction[1] = 'S'
  elif (old_posY > new_posY): #moved to the north
    direction[1] = 'N'

  return direction

gameName = sys.argv[1] # INSERT GAME NAME HERE
playerName = sys.argv[2]
n_steps = 9

# Establish Connection
dumb_ch = CommunicationHandler(host, serverPort)
print(dumb_ch.host)

# Join the Game
join = dumb_ch.joinGame(gameName, playerName, train=False)
print(join)

map = dumb_ch.getMap()
status = dumb_ch.getStatus()
myPlayer = extractMyInfo(status)
players = extractPlayerInfo(status)
dumb_chat = Chat(host, chatPort, myPlayer)
dumb_chat.join(gameName, myPlayer.team)
chat_sleep = random.randint(5,15)
chat_timer = time.time()
read_timer = time.time()
chat_messages = []

# Status requests until the game is started
judged_players = []
written = False
while(gameState(status) == 'LOBBY'):
  judged_players = judgePlayers(judged_players, players, dumb_ch)
  status = dumb_ch.getStatus()
  players = extractPlayerInfo(status)
  if (not written and (time.time() - chat_timer) >= chat_sleep):
    dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['salute']))
    written = True



# Game has started
startGame = time.time()
chat_timer = startGame
chat_sleep = random.randint(10,30)
players = extractPlayerInfo(status)
myPreparedMap = prepareMap3(copy.deepcopy(map), myPlayer, players, size=(len(map),len(map[0]))) #pos with shooting lines

# function to calculate how close each player is to the flags
myFlagY, myFlagX = findOnMap(map, myPlayer.myFlag)
enemyFlagY, enemyFlagX = findOnMap(map, myPlayer.enemyFlag)

# Game loop
while (gameState(status) == 'ACTIVE'):
  minPath = 9999
  for player in players.values():
    if (player.team != myPlayer.loyalty and player.alive): #impostor
      player_dist = distance(int(player.posX), int(player.posY), myFlagX, myFlagY)
      player.setFlagDistance(player_dist)
      if(minPath > player_dist):
        minPath = player_dist
        closer_player = player

  posY, posX = myPlayer.posY, myPlayer.posX
  # use zones to move to the safest one for the first 5sec
  elapsed_time = time.time() - startGame
  while (elapsed_time < 4.5): #cannot shoot enemies -> go to a safe place closer to the flag
    posY, posX = myPlayer.posY, myPlayer.posX
    zones, best_pos = computeDangerMap(myPreparedMap, enemyFlagX, enemyFlagY)
    myZone = findZone(posX, posY)
    best, bestX, bestY = bestAdjacientZone(zones, best_pos, posX, posY)
    zone_path = findPath(myPreparedMap, posX, posY, bestX, bestY)
    p = pathToCommand(dumb_ch, zone_path)
    # update the status to get the update info
    status = dumb_ch.getStatus()
    myPlayer = extractMyInfo(status)
    players = extractPlayerInfo(status)
    players = updatePlayerPosition(status, players)
    map = updateMapFromStatus(map, myPlayer, players)
    elapsed_time = time.time() - startGame
    print('Elapsed time: ' + str(elapsed_time))

  # for each player, check the (x,y) distance from me and see if i am in shooting range
  n_steps = calculateSteps(players, myPlayer)
  if (n_steps == 0 and (time.time() - chat_timer) >= chat_sleep):
    chat_timer = time.time()  
    chat_sleep = random.randint(10,30)
    dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['kill']))

  #--------impostor case -> run to the "enemy flag" and shoot all your friends
  if (myPlayer.team != myPlayer.loyalty):
    myPreparedMap = prepareImpostorMap(copy.deepcopy(map), myPlayer, players)
    posY, posX = myPlayer.posY, myPlayer.posX
    zones, best_pos = computeImpostorDangerMap(myPreparedMap, players)
    myZone = findZone(posX, posY)
    best, bestX, bestY = bestAdjacientZone(zones, best_pos, posX, posY)
    if (zones[myZone[1]][myZone[0]] > best):
      zone_path = findPath(myPreparedMap, posX, posY, bestX, bestY)
      p = pathToCommand(dumb_ch, zone_path[0:n_steps+1])
    if ((time.time() - chat_timer) >= chat_sleep):
      chat_timer = time.time()  
      chat_sleep = random.randint(10,30)
      dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['defend']))
    defendPosition(dumb_ch, map, players, myPlayer)


  #--------true loyalty
  else:
    # extract info from status to avoid calling getMap()
    myPreparedMap = prepareMap3(copy.deepcopy(map), myPlayer, players, size=(len(map),len(map[0]))) #pos with shooting lines
    posY, posX = myPlayer.posY, myPlayer.posX
    n_steps = calculateSteps(players, myPlayer)
    print('n_steps = ' + str(n_steps))
    myPath = findPath(myPreparedMap, posX, posY, enemyFlagX, enemyFlagY)
    if (len(myPath) > 0):
      if ((time.time() - chat_timer) >= chat_sleep):
        chat_timer = time.time()  
        chat_sleep = random.randint(10,30)
        dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['attack']))
      p = pathToCommand(dumb_ch, myPath[0:n_steps+1])

    elapsed_time = time.time() - startGame

    if ((time.time() - chat_timer) >= chat_sleep):
      chat_timer = time.time()  
      chat_sleep = random.randint(10,30)
      dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['kill']))

    defendPosition(dumb_ch, map, players, myPlayer, iter=3)

    status = dumb_ch.getStatus()
    myPlayer = extractMyInfo(status)
    players = extractPlayerInfo(status)
    players = updatePlayerPosition(status, players)
    map = updateMapFromStatus(map, myPlayer, players)
    myPreparedMap = prepareMap3(copy.deepcopy(map), myPlayer, players, size=(len(map),len(map[0])))

    if (myPlayer.energy == 0): # run to the flag since you have no energy
      myPath = findPath(myPreparedMap, posX, posY, enemyFlagX, enemyFlagY)
      if ((time.time() - chat_timer) >= chat_sleep):
        chat_timer = time.time()  
        chat_sleep = random.randint(10,30)
        dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['attack']))
      if (len(myPath) > 0):
        p = pathToCommand(dumb_ch, myPath) 

# Game is over
myScore = extractMyInfo(status).score
print(myScore)

# Posting on chat
if (myScore > 40):
  dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['victory']))
else:
  dumb_chat.post(gameName, random.choice(CHAT_DICTIONARY['defeat']))