from collections import defaultdict, deque
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
from prettytable import PrettyTable
from copy import deepcopy

LAMBDA = 'lambda'
DELTA = 'delta'
INPUT = 'input'
OUTPUT = 'output'

import graphviz  # type: ignore

states = ['hungry', 'satisfied', 'full', 'sick']
q0 = 'full'
F = ['full', 'hungry']


def saveFSM(
    vertices: Set[str], edges: List[Tuple[str, str, str]], q0: str,
    F: Set[str], fileName: str
  ):
  dot = graphviz.Digraph(comment='FSM', filename=fileName)

  q_attr = {'shape': 'circle'}
  for v in vertices:
    if v != q0 and v not in F:
      dot.node(v)  # type: ignore

  F_attr = {'shape': 'doublecircle'}
  dot.attr('node', **F_attr)  # type: ignore
  for f in F:
    if f != q0:
      dot.node(f)  # type: ignore

  q0_attr = {'shape': 'circle', 'color': 'aquamarine', 'style': 'filled'}
  if q0 in F:
    q0_attr['shape'] = 'doublecircle'
  dot.attr('node', **q0_attr)  # type: ignore
  dot.node(q0)  # type: ignore

  dot.attr('node', **q_attr)  # type: ignore
  for (u, v, label) in edges:
    dot.edge(u, v, label)  # type: ignore
  dot.render(format='svg').replace('\\', '/')  # type: ignore


def err(message: str):
  raise Exception(message)


def printTriangle(triangle: list[list[int]]):
  print(
    '\n'.join([' '.join([str(tij) for tij in ti])
    for ti in triangle]) + '\n-------'
    )


class Node:

  def __init__(self, parent: int = -1, rank: int = -1) -> None:
    self.parent = parent
    self.rank = rank


class UnionFind:

  def __init__(self, size: int) -> None:
    self.islands = [Node(i, 0) for i in range(size)]
    self.count = size

  def find(self, i: int):
    if self.islands[i].parent != i:
      self.islands[i].parent = self.find(self.islands[i].parent)
    return self.islands[i].parent

  def union(self, i: int, j: int):
    i_par = self.find(i)
    j_par = self.find(j)
    if i_par == j_par:
      return
    if self.islands[i_par].rank < self.islands[j_par].rank:
      self.islands[i_par].parent = j_par
    elif self.islands[i_par].rank > self.islands[j_par].rank:
      self.islands[j_par].parent = i_par
    else:
      self.islands[j_par].parent = i_par
      self.islands[i_par].rank += 1
    self.count -= 1


class DFA:

  def __init__(self, Sigma: Set[str], vizFile: Path) -> None:
    self.q: Set[str] = set()
    self.Sigma = Sigma
    self.delta: defaultdict[str, defaultdict[str,
      str]] = defaultdict(lambda: defaultdict(str))
    self.q0: str
    self.F: Set[str] = set()
    self.vizFile = vizFile

  def _delta2table(self):
    q = list(sorted(self.q))
    Sigma = list(sorted(self.Sigma))
    table = PrettyTable()
    table.field_names = [DELTA] + Sigma
    for qi in q:
      if qi not in self.delta:
        table.add_row([qi] + [" "] * len(Sigma))  # type: ignore
        continue
      row = [qi]
      for ch in Sigma:
        if ch not in self.delta[qi]:
          row.append(" ")
          continue
        row.append(self.delta[qi][ch])
      table.add_row(row)  # type: ignore
    return table

  def __repr__(self) -> str:
    res = f'''
q: {list(sorted((self.q)))}
Sigma: {self.Sigma}
q0: {self.q0}
F: {list(sorted((self.F)))}
delta:\n{self._delta2table()}
'''
    return res

  def vizualize(self):
    vertices = self.q
    edges: List[Tuple[str, str, str]] = []
    q0 = self.q0
    F = self.F
    qi2qj: defaultdict[str, defaultdict[str,
      Set[str]]] = defaultdict(lambda: defaultdict(lambda: set()))
    for qi in self.delta:
      for ch in self.delta[qi]:
        qj = self.delta[qi][ch]
        qi2qj[qi][qj].add(ch)
    for qi in qi2qj:
      for qj in qi2qj[qi]:
        edges.append((qi, qj, ', '.join(list(sorted(qi2qj[qi][qj])))))
    fileName = str(self.vizFile)
    saveFSM(vertices, edges, q0, F, fileName)

  def simulate(self, inp: str | list[str]):
    q = self.q0
    res = f'{q}'
    for ch in inp:
      if ch not in self.delta[q]:
        res += " REJECT"
        return res
      q = self.delta[q][ch]
      res = f'{res} -- {ch} --> {q}'
    if q in self.F:
      res += " ACCEPT"
    else:
      res += " REJECT"
    return res

  def minimize(self):

    qNames = list(sorted(self.q))
    q2Name: defaultdict[int, str] = defaultdict(lambda: "")
    for idx in range(len(qNames)):
      q2Name[idx] = qNames[idx]
    name2q = {q2Name[i]: i for i in range(len(qNames))}

    delta: defaultdict[int, defaultdict[str,
      Set[int]]] = defaultdict(lambda: defaultdict(set))
    for qi in self.delta:
      for ch in self.delta[qi]:
        qj = self.delta[qi][ch]
        delta[name2q[qj]][ch].add(name2q[qi])

    triangle = [[0] * i for i in range(len(qNames))]
    for i in range(len(qNames)):
      for j in range(i + 1, len(qNames)):
        if not ((qNames[i] in self.F) == (qNames[j] in self.F)):
          triangle[j][i] = 1
          continue

    print(
      '+++ Map of state number to state name serving as the legend in triangle:\n\n'
      )
    print('\n'.join([f'{k} | {q2Name[k]}' for k in q2Name.keys()]))
    print(
      '\n\n+++ Marked all pairs of states where one is a final state, while the other is not'
      )
    printTriangle(triangle)
    while True:
      work = False
      curr = deepcopy(triangle)
      for i in range(len(qNames)):
        for j in range(i + 1, len(qNames)):
          if curr[j][i] > 0:
            continue
          for ch in self.Sigma:
            i_next = name2q[self.delta[q2Name[i]][ch]]
            j_next = name2q[self.delta[q2Name[j]][ch]]
            if i_next == -1 or j_next == -1 or i_next == j_next:
              continue
            i_next, j_next = list(sorted([i_next, j_next]))
            if curr[j_next][i_next] == 0:
              continue
            triangle[j][i] = curr[j_next][i_next] + 1
            work = True
            break
      if not work:
        break
    print('\n\n+++ Triangle filled')
    printTriangle(triangle)

    islands = UnionFind(len(qNames))
    for i in range(len(qNames)):
      for j in range(i + 1, len(qNames)):
        if triangle[j][i] == 0:
          islands.union(i, j)

    minimized = deepcopy(self)
    minimized.vizFile = self.vizFile.parent.joinpath('dfa_minimized.gv')
    if islands.count == len(qNames):
      return minimized

    qMinIdcs = list(sorted(set([islands.find(i) for i in range(len(qNames))])))
    minimized.q = set([qNames[idx] for idx in qMinIdcs])
    for i in range(len(qNames)):
      if i not in qMinIdcs:
        minimized.delta.pop(qNames[i])
    for qi in minimized.delta:
      for ch in minimized.delta[qi]:
        minimized.delta[qi][ch] = qNames[islands.find(
          name2q[minimized.delta[qi][ch]]
          )]
    minimized.q0 = qNames[islands.find(name2q[minimized.q0])]
    F: Set[str] = set()
    for f in minimized.F:
      F.add(qNames[islands.find(name2q[f])])
    minimized.F = F  # type: ignore
    return minimized


def serializeStateSet(states: Set[str]):
  return str(list(sorted(states)))


class NFA:

  def __init__(
      self, q: List[str], Sigma: List[str], delta: Dict[str, Dict[str,
    List[str]]], q0: str, F: List[str], vizFile: Path
    ) -> None:
    self.q = set(q)
    self.Sigma = set(Sigma)
    self.delta: defaultdict[str, defaultdict[str,
      Set[str]]] = defaultdict(lambda: defaultdict(lambda: set()))
    # Convert the list of next states for each transition rule into a set.
    for qi in delta:
      for ch in delta[qi]:
        self.delta[qi][ch] = set(delta[qi][ch])
    self.q0 = q0
    self.F = set(F)
    self.vizFile = vizFile

  def _delta2table(self):
    q = list(sorted(self.q))
    Sigma = list(sorted(self.Sigma) + [LAMBDA])
    table = PrettyTable()
    table.field_names = [DELTA] + Sigma
    for qi in q:
      if qi not in self.delta:
        table.add_row([qi] + [" "] * len(Sigma))  # type: ignore
        continue
      row = [qi]
      for ch in Sigma:
        if ch not in self.delta[qi]:
          row.append(" ")
          continue
        row.append(str(self.delta[qi][ch]))
      table.add_row(row)  # type: ignore
    return table

  def __repr__(self) -> str:
    res = f'''
q: {list(sorted((self.q)))}
Sigma: {self.Sigma}
q0: {self.q0}
F: {list(sorted((self.F)))}
delta:\n{self._delta2table()}
'''
    return res

  def vizualize(self):
    vertices = self.q
    edges: List[Tuple[str, str, str]] = []
    q0 = self.q0
    F = self.F
    qi2qj: defaultdict[str, defaultdict[str,
      Set[str]]] = defaultdict(lambda: defaultdict(lambda: set()))
    for qi in self.delta:
      for ch in self.delta[qi]:
        for qj in self.delta[qi][ch]:
          qi2qj[qi][qj].add(ch)
    for qi in qi2qj:
      for qj in qi2qj[qi]:
        edges.append((qi, qj, ', '.join(list(sorted(qi2qj[qi][qj])))))
    fileName = str(self.vizFile)
    saveFSM(vertices, edges, q0, F, fileName)

  def transition(self, q: str, ch: str):
    return self.delta[q][ch]

  def serializeStateSet(self, states: Set[str]):
    return str(list(sorted(states)))

  def lambdaClosure(self, states: Set[str]):
    currState = states
    while True:
      nextState: Set[str] = currState
      for state in currState:
        nextState = nextState.union(self.transition(state, LAMBDA))
      if nextState == currState:
        return nextState
      currState = nextState

  def _hasAcceptState(self, states: Set[str]):
    return len(self.F.intersection(states)) > 0

  def toDFA(self):
    dfa = DFA(self.Sigma, self.vizFile.parent.joinpath('dfa_converted.gv'))

    # Determining the set of possible initial states
    seed = self.lambdaClosure({self.q0})

    dfa.q0 = serializeStateSet(seed)
    dfa.q.add(dfa.q0)

    Sigma = list(sorted(self.Sigma))
    queue: deque[Set[str]] = deque([seed])
    while queue:
      currState = queue.popleft()
      qi = serializeStateSet(currState)
      if self._hasAcceptState(currState):
        dfa.F.add(qi)
      currState = list(sorted(currState))
      for ch in Sigma:
        nextState: Set[str] = set()
        for q in currState:
          nextState = nextState.union(
            self.lambdaClosure(self.transition(q, ch))
            )
        qj = serializeStateSet(nextState)
        if qj not in dfa.q:
          queue.append(nextState)
          dfa.q.add(qj)
        dfa.delta[qi][ch] = qj
    return dfa


if __name__ == '__main__':

  if len(sys.argv) not in [2, 3]:
    print(
      'Usage: python dfa.py <path-to-nfa-spec> [<path-to-input-strings-for-simulation>]'
      )

  nfaPath = sys.argv[1]
  if not os.path.isfile(nfaPath):
    err(f'Could not find the file "{nfaPath}"')

  vizFile = Path(nfaPath).parent.absolute()

  nfaJson = json.load(open(nfaPath))
  nfa = NFA(vizFile=vizFile.joinpath('nfa.gv'), **nfaJson)
  print('\n\n*** Provided NFA')
  print(nfa)
  nfa.vizualize()
  dfa = nfa.toDFA()
  print('\n\n*** Converted DFA')
  print(dfa)
  dfa.vizualize()
  minimized = dfa.minimize()
  print('\n\n*** Minimized DFA')
  print(minimized)
  minimized.vizualize()

  if len(sys.argv) == 2:
    while True:
      inp = input("Enter the string to simulate. Press enter to exit: ")
      if not inp:
        break
      print(minimized.simulate(inp))
  else:
    inpPath = sys.argv[2]
    if not os.path.isfile(nfaPath):
      err(f'Could not find the input file, can\'t simulate "{nfaPath}"')
    inpList: List[Dict[str, str]] = json.load(open(inpPath))
    for inp in inpList:
      fsmInp = inp[INPUT]
      print(f'Input string: {fsmInp}')
      fsmOut = minimized.simulate(fsmInp)
      print(fsmOut)
