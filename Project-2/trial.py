import numpy as np 
def to_binary(n):
  bin_arr = [0,0,0,0]
  i = 0
  while (n>0):
      bin_arr[i] = n%2
      n = int(n/2)
      i = i+1
  bin_arr.reverse()
  return np.array(bin_arr)

def to_decimal(arr):
  bin_arr = list(arr)
  bin_arr.reverse()
  dec_val = 0
  for i in range(len(bin_arr)):
    dec_val = dec_val + bin_arr[i] * (2**i)
  
  return dec_val

val = np.array([0,1,1,1])
print(type(to_binary(5)))
print(to_decimal(val))