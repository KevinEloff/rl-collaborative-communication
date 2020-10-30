import numpy as np
import re
from variables import Actions, distances
import stringhandling as sh

def get_action_sequence(state):
    action_seq = state_to_actions(state, limit_n=5)
    words_seq = actions_to_words(action_seq)
    actual_action_seq = words_to_actions(words_seq)

    # actual_action_seq = action_seq
    return actual_action_seq, words_seq, action_seq

def words_to_actions(words_sequence):
    s = ' '.join(words_sequence)
    s.replace('and then', 'and_then')
    seqs = re.split(' and_then | and | then ',s)


    actions = []

    for seq in seqs:
        seq = sh.filtertext(seq, remove_words)
        cmd_type = getCmd(seq)
        if cmd_type == 0:
            direction = getDir(seq)
            dist = getDist(seq)
#             print(direction)
            if direction == -1 or dist == -1:
                continue
            
            actions += [Actions(direction)]*dist

    return actions

def getCmd(seq):
    words = seq.split(' ')
    #Quenstion type in list
    index = -1
    for word in words[0:1]:
        num = sh.bestmatch(commands, word, threshold=0.1, max=2)
        if index == -1 and num != -1:
            index = num
        elif num != -1:
            return -1

    return index

def getDir(seq):
    words = seq.split(' ')
    #Quenstion type in list
    index = -1
    for word in words[1:]:
#         print(word)
        num = sh.bestmatch(moves, word, threshold=0.05, max=0)
#         print(num)
        if index == -1 and num != -1:
            index = num
        elif num != -1:
            return -1

    return index

def getDist(seq):
    dist = -1
    for i,j in distances.items():
        if i in seq[1:-1] or str(j) in seq[1:-1]:
            if dist == -1:
                dist = j
                if seq.count(i) + seq.count(str(j)) > 1:
                    return -1
            else:
                return -1

    return dist

def state_to_actions(state, limited=True, limit_n=0, straight_only=True):
    start = {
        'y': state['agent'][0], 
        'x': state['agent'][1],
        'prev': []
    }

    q = [start]
    seen = [[start['y'], start['x']]]
    while len(q) > 0:
        block = q.pop(0)

        if np.all([block['y'], block['x']]==state['target']):
            break

        for action in Actions:
            if action.value == 4: continue
            
            dx = np.sin(action.value * np.pi/2).astype(int)
            dy = -np.cos(action.value * np.pi/2).astype(int)
            
            if block['y'] + dy < 0 or block['y'] + dy >= state['grid'].shape[0]:
                continue
            if block['x'] + dx < 0 or block['x'] + dx >= state['grid'].shape[-1]:
                continue
            if [block['y']+dy, block['x']+dx] in seen:
                continue
            if state['grid'][block['y']+dy, block['x']+dx]:
                continue
            q.append({
                'y': block['y'] + dy, 
                'x': block['x'] + dx,
                'prev': block['prev'] + [action]
            })
            seen += [[block['y'] + dy, block['x'] + dx]]
    
    if not np.all([block['y'], block['x']]==state['target']):
        return []
    
    if not limited:
        return block['prev']

    if limit_n != 0:
        if not straight_only:
            return block['prev'][:limit_n]
        block['prev'] = block['prev'][:limit_n]

    count_same = 0
    for a in block['prev']:
        if block['prev'][0] == a: count_same += 1
        else: break
    
    return [block['prev'][0]]*count_same



def get_sequence(action, dist):
    start_options = ['move', 'go', 'walk', 'proceed']
    start = start_options[np.random.randint(0, len(start_options))]

    step_options = ['block', 'step', 'space']
    step = step_options[np.random.randint(0, len(step_options))]
    if dist > 1: step += 's'

    direction = [['north', 'up'], ['east', 'right'], ['south', 'down'], ['west', 'left']][action.value][np.random.randint(0,2)]
    
    dist = dist
    if np.random.randint(0,2):
        for i,j in distances.items():
            if j == dist:
                dist = i
    else:
        dist = str(dist)

    seq = []
    if start is not None:
        seq += [start]

    if np.random.randint(0,1):
        seq += [direction, dist, step]
    else:
        seq += [dist, step, direction]
    
    return seq

def actions_to_words(action_sequence):
    if len(action_sequence) == 0:
        return []
    word_sequence = []

    prev_action = None
    action_count = 1
    for action in action_sequence:
        if prev_action == action:
            action_count += 1
        elif prev_action != None:
            if len(word_sequence) != 0:
                word_sequence += [['then'], ['and','then'], ['and']][np.random.randint(0,3)]
            word_sequence += get_sequence(prev_action, action_count)
            action_count = 1
        prev_action = action

    if len(word_sequence) != 0:
        word_sequence += [['then'], ['and','then'], ['and']][np.random.randint(0,3)]
    word_sequence += get_sequence(action, action_count)

    return word_sequence



# known_text = [
#     ['blocks north', 'blocks up', 'steps north', 'steps up', 'space north', 'space up'],
#     ['blocks east', 'blocks right', 'steps east', 'steps right', 'space east', 'space right'],
#     ['blocks south', 'blocks down', 'steps south', 'steps down', 'space south', 'space down'],
#     ['blocks west', 'blocks left', 'steps west', 'steps left', 'space west', 'space left'],
# ]
remove_words = ['a','is','that','this','the','i','to','lol','it','again','are', 'must']
commands = [
    ['move', 'go', 'walk', 'proceed']
]
moves = [
    ['north','up'],
    ['east','right'],
    ['south','down'],
    ['west','left']
]