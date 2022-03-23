### Usage

```sh
# Install graphviz. ref: https://graphviz.org/download/
# Linux
sudo apt install graphviz
# Mac
brew install graphviz
pip install -r requirements.txt
python3 dfa.py input/54a/definition.json input/54a/strings.json
```

```
25c: (abc)*a*
25d: (ba+bb)*+(aa+ab)*
28: (ab+ba)* + (ab+ba)*((aa(a+b)*bb) + (bb(a+b)*aa))(a+b)*
29: (a + b)* b (a + b) ( a + b )
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
