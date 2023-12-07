PLAN FOR ICML24 PARER

Problems:
1.	The connection between the score learned by Score Matching and the bridge simulation (forward and backward), how to exactly use the learned score to simulate the bridge?
2.	Why do we train the denoising score matching between individual delta t, and then it can represent the whole transition process? (perhaps also consider the Ito and Stratonovich discretization)
3.	The covariance matrix of infinite dimensional noise, how to compute it correctly when discretizing into finite grid?
4.	*Directly learn the score for infinite dimensional grid
5.	*A code library/repository 

Timetable (11 weeks from now on)
W1: Read “Simulating Diffusion Bridges with Score Matching” paper for both Libby and Gefan, Libby learns about the score matching, Gefan learns about Ito and Stratonovich integral and something else about SDEs (1 week) (Done)

W2: Tackle Problem 2, some programming experiments (e.g. test it for some simple linear and nonlinear SDEs) (2 weeks)
•	For B.M., compare everything learned with the ground truth (Gaussians) (done)
•	Increase the dimensionality. 
•	Why do we need Z* for learning X*? (done)
•	Start with the fixed Q matrix evaluated at X0.
•	Make Q varies with t.

W4: Tackle Problem 1, some programming experiments (e.g. simulate some simple bridges use learned score) (2 weeks)
•	Libby: write up maths finite vs infinite, stratonovich loss function, look at backward backward code
•	Gefan: Make backward backward work. Increase the complexity. Check diffrax library

W6: Know well about Problem 1&2, simulate some finite dimensional SDEs (especially non-linear ones, but know exactly the covariance in advance) (1 week)

W7: Solve problem 3, plug in the correct covariance matrix for the landmark SDE, get some results and figures for the paper. (1 week)

W8-11: Paper time, integrate the codes. (3 weeks)
