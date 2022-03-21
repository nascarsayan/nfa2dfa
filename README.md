### Usage

```sh
pip install -r requirements.txt
python3 dfa.py input/54a.txt
```

### Problem statement
Assignment 3: DFA conversion algorithms (due by email and hard copy 03/23/22) 

Part I. Implement Algorithm 5.6.3 from Sudkamp to convert an NFA-lambda M = (Q, Sigma,delta,q0,F) into a DFA M' = DM.

Part II. Implement the DFA minimization algorithm from the HMU text, to convert DFA M' to a DFA M" with minimal number of states. Read HMU handout: Ch. 4 Sections 4.4.1-3. 

Part III. Write a DFA simulator to generate the computation of the DFA on a given string. 

The program should work for a general (user-specified) NFA-lambda as input (not just for specific examples), which should be read from a file. 
Parts I, II and Ill have to be coded into one program (not three separate programs). 
The input (string) for Part III may be given interactively. 

Submit all code in a zip or tar archive, by email. Submit your report by email and hard copy. Be ready to execute and demonstrate your program for different NFA-lambdas and strings.

Convert a given NFA-lambda M = (Q, Sigma, delta, q0, F) into a DFA M' = DM; then minimize M', resulting in a minimized DFA M".
Include the "t-table" in the output from part I, as well as the transition table and full specification of the obtained DFA (M).
Include the triangular table (of distinguishabilities) in your output from part II.
Test for given input NFA-lambda's M:
Derive NFA-lambdas for the exercises of Sudkamp Ch. 5, p. 186 #25 (c,d), p. 187 ##28, 29. 
For each problem, list 
- the input specification of the NFA-lambda, and 
- the input and outputs of the resulting (minimized) DFA for two strings that should be accepted and two strings that should be rejected.
For each string, print the computation (i.e., the sequence of configurations) of the DFA on the string, followed by "Accepted" or "Rejected". Submit a report including a description of your design and implementation for parts I, II and III, and the sample problems with inputs and requested outputs.



### Note to the client

The compensation and the time required / provided for the assignment aren't directly proportional.
I've coded the NFA -> DFA completely myself, and it was interesting.
I will need some more time for the other tasks.

#### Todo
II. DFA minimization [reference](https://github.com/navin-mohan/dfa-minimization/blob/master/dfa.py)

III. Simulation
Will try to complete this.
