from collections import defaultdict, deque
import json
import os
import sys
from typing import Deque, Dict, List, Set, Tuple
from prettytable import PrettyTable
from copy import deepcopy

LAMBDA = 'lambda'
DELTA = 'delta'


def err(message: str):
  raise Exception(message)


class DFARev:

  def __init__(
      self, q2Name: defaultdict[int, str], q: Set[int], Sigma: Set[str],
      delta: defaultdict[int, defaultdict[str, Set[int]]]
    ) -> None:
    self.q2Name = q2Name
    self.q = q
    self.Sigma = Sigma
    self.delta = delta
    # self.q0: List[int] = []

  def getNextPairs(self, pair: Tuple[int, int], ch: str):
    qi_next = set(filter(lambda x: x != -1, self.delta[pair[0]][ch]))
    qj_next = set(filter(lambda x: x != -1, self.delta[pair[1]][ch]))
    pairs = set(tuple(sorted(pi)) for pi in set(zip(qi_next, qj_next)))
    pairs = set(filter(lambda x: x[0] != x[1], pairs))
    return pairs


class DFA:

  def __init__(self, Sigma: Set[str]) -> None:
    self.q: Set[str] = set()
    self.Sigma = Sigma
    self.delta: defaultdict[str, defaultdict[str,
      str]] = defaultdict(lambda: defaultdict(str))
    self.q0: str
    self.F: Set[str] = set()

  def _delta2table(self):
    q = list(sorted(self.q))
    Sigma = list(sorted(self.Sigma))
    table = PrettyTable()
    table.field_names = [DELTA] + Sigma
    for qi in q:
      if qi not in self.delta:
        table.add_row([qi] + [" "] * len(Sigma))  # type: ignore
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

    rev = DFARev(q2Name, set(range(len(qNames))), self.Sigma, delta)
    triangle = [[0] * len(qNames) for _ in qNames]
    queue: Deque[Tuple[int, int, int]] = deque()
    for i in range(len(qNames)):
      for j in range(i + 1, len(qNames)):
        if not ((qNames[i] in self.F) == (qNames[j] in self.F)):
          queue.append((i, j, 1))
          triangle[j][i] = 1

    while queue:
      (i, j, dist) = queue.popleft()
      for ch in self.Sigma:
        nextPairs = rev.getNextPairs((i, j), ch)
        for (i_prev, j_prev) in nextPairs:
          if triangle[j_prev][i_prev] > 0:
            continue
          queue.append((i_prev, j_prev, dist + 1))
          triangle[j_prev][i_prev] = dist + 1
        # print(
        #   '\n'.join([' '.join([str(tij) for tij in ti])
        #   for ti in triangle]) + '\n-------'
        #   )

    print('\n'.join([' '.join([str(tij) for tij in ti]) for ti in triangle]))

    collapsed: Set[int] = set()
    for i in range(len(rev.q)):
      for j in range(i + 1, len(rev.q)):
        if triangle[j][i] == 0:
          collapsed = collapsed.union([i, j])

    minimized = deepcopy(self)
    if len(collapsed) == 0:
      return minimized

    mergedState = list(sorted(collapsed))[0]
    discarded = collapsed.difference([mergedState])
    qMinIdcs = list(sorted(rev.q.difference(discarded)))
    minimized.q = set([qNames[i] for i in qMinIdcs])
    for i in discarded:
      minimized.delta.pop(qNames[i])
    for qi in minimized.delta:
      for ch in minimized.delta[qi]:
        if name2q[minimized.delta[qi][ch]] in discarded:
          minimized.delta[qi][ch] = qNames[mergedState]
    if name2q[minimized.q0] in discarded:
      minimized.q0 = qNames[mergedState]
    F: Set[str] = set()
    for f in minimized.F:
      if name2q[f] in discarded:
        F.add(qNames[mergedState])
      else:
        F.add(f)
    minimized.F = F  # type: ignore
    return minimized

  def minimizeBak(self):
    rev = deepcopy(self)
    delta: defaultdict[str, defaultdict[str,
      str]] = defaultdict(lambda: defaultdict(str))
    for qi in self.delta:
      for ch in self.delta[qi]:
        qj = self.delta[qi][ch]
        delta[qj][ch] = qi
    rev.delta = delta

    q = list(sorted(rev.q))
    q2idx: defaultdict[str, int] = defaultdict(lambda: -1)
    for idx in range(len(q)):
      q2idx[q[idx]] = idx
    triangle = [[0] * len(q) for _ in q]
    queue: Deque[Tuple[int, int, int]] = deque()
    for i in range(len(q)):
      for j in range(i + 1, len(q)):
        if not ((q[i] in self.F) == (q[j] in self.F)):
          queue.append((i, j, 1))
          triangle[j][i] = 1

    while queue:
      (i, j, dist) = queue.popleft()
      for ch in self.Sigma:
        i_prev = q2idx[rev.delta[q[i]][ch]]
        j_prev = q2idx[rev.delta[q[j]][ch]]
        if (i_prev < 0) or (j_prev < 0) or (i_prev == j_prev):
          continue
        i_prev, j_prev = list(sorted([i_prev, j_prev]))
        if triangle[j_prev][i_prev] > 0:
          continue
        queue.append((i_prev, j_prev, dist + 1))
        triangle[j_prev][i_prev] = dist + 1
      # print(
      #   '\n'.join([' '.join([str(tij) for tij in ti])
      #   for ti in triangle]) + '\n-------'
      #   )

    print('\n'.join([' '.join([str(tij) for tij in ti]) for ti in triangle]))

    collapsed: Set[int] = set()
    for i in range(len(q)):
      for j in range(i + 1, len(q)):
        if triangle[j][i] == 0:
          collapsed = collapsed.union([i, j])

    minimized = deepcopy(self)
    if len(collapsed) == 0:
      return minimized

    mergedState = list(sorted(collapsed))[0]
    discarded = collapsed.difference([mergedState])
    qMinIdcs = list(sorted(set(list(range(len(q)))).difference(discarded)))
    minimized.q = set([q[i] for i in qMinIdcs])
    for i in discarded:
      minimized.delta.pop(q[i])
    for qi in minimized.delta:
      for ch in minimized.delta[qi]:
        if q2idx[minimized.delta[qi][ch]] in discarded:
          minimized.delta[qi][ch] = q[mergedState]
    if q2idx[minimized.q0] in discarded:
      minimized.q0 = q[mergedState]
    F: Set[str] = set()
    for f in minimized.F:
      if q2idx[f] in discarded:
        F.add(q[mergedState])
      else:
        F.add(f)
    minimized.F = F  # type: ignore
    return minimized


def serializeStateSet(states: Set[str]):
  return str(list(sorted(states)))


class NFA:

  def __init__(
    self,
    q: List[str],
    Sigma: List[str],
    delta: Dict[str, Dict[str, List[str]]],
    q0: str,
    F: List[str],
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

  def _delta2table(self):
    q = list(sorted(self.q))
    Sigma = list(sorted(self.Sigma) + [LAMBDA])
    table = PrettyTable()
    table.field_names = [DELTA] + Sigma
    for qi in q:
      if qi not in self.delta:
        table.add_row([qi] + [" "] * len(Sigma))  # type: ignore
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
    dfa = DFA(self.Sigma)

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
  nfaPath = '/home/ubuntu/Code/pad/dfa/input/54a.json'
  if len(sys.argv) > 1:
    nfaPath = sys.argv[1]
  if not os.path.isfile(nfaPath):
    err(f'Could not find the file "{nfaPath}"')

  nfaJson = json.load(open(nfaPath))
  nfa = NFA(**nfaJson)
  print(nfa)
  dfa = nfa.toDFA()
  print(dfa)
  minimized = dfa.minimize()
  print(minimized)
  minimized = dfa.minimizeBak()
  print(minimized)
