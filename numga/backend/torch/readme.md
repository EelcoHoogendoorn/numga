NOTE: This torch backend has not been used much or been tested a lot.

Without jitting, running GA-expression in torch appears to be about equally attractive,
as running them in plain numpy. 
That is, not very attractive; since we tend to be dealing with large computation graphs 
of relatively low compute intensity.