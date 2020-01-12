from math import *

#converts a number 'num' to a binary string of length n
def bin_len_n(num, n):
	b = str(bin(num))[2:]
	while len(b)<n:
		for j in range(n-len(b)):
			b = '0' + b
	return b

#creates the vector space of F2^n, returns all possible elements as a list of strings of 0's and 1's
def Vn(n):
	Vn = []			
	for i in range(2**n):
		Vn.append(bin_len_n(i,n))
	return Vn


#f = [1,0,1,1,0,1,0,0]	#truth table of f


#returns a list of the terms in the ANF, ordered by degree. if ANF=0, returns empty string
def ANF(f):
	n = int(log(len(f),2))
	C = []				#coefficients of ANF

	for i in range(len(Vn(n))):
		c=0
		for j in range(i+1):
			for k in range(n):
				truth =0
				if Vn(n)[i][k]<Vn(n)[j][k]:
					truth = 1
					break
				else:
					continue
			if truth == 0:
				c+=f[j]
		C.append(c%2)

	terms = []
	for i in range(2**n):
		term = ''
		if C[i]==1:
			if i==0:
				terms.append('1')
			for j in range(n):
				if Vn(n)[i][j]=='1':
					term+='x'+str(j+1)
			terms.append(term)
	return C, sorted(terms, key=len)

#def ANF(f):
#	n = int(log(len(f),2))
#	C = []				#coefficients of ANF
#
#	for i in range(len(Vn(n))):
#		c=0
#		for j in range(i+1):
#			for k in range(n):
#				truth =0
#				if Vn(n)[i][k]<Vn(n)[j][k]:
#					truth = 1
#					break
#				else:
#					continue
#			if truth == 0:
#				c+=f[j]
#		C.append(c%2)
#    return Vn(n)[C]

def degree(f):
	return len(ANF(f)[-1])/2

if __name__ == '__main__':
    f = [1,0,1,1,1,0,0,1]
    print(ANF(f), degree(f))
			

