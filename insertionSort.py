# Implementation of Insertion Sort

def insertion_sort(arr):
	""" Returns a Sorted Array"""

	for i in range(1,len(arr)):
		#Insert this appropriately into the Unsorted array [0:i]
		unsortedElement = arr[i]
		j = i-1
		while j>=0 and arr[j] > unsortedElement:
			#swap 
			arr[j+1] = arr[j]
			j-=1
		arr[j+1] = unsortedElement
	return arr


def merge_sort(arr):

	if len(arr)>1:

		mid = len(arr)//2
		lefthalf = arr[:mid]
		righthalf = arr[mid:]
		print("Left Half is:"+str(lefthalf))
		print("Right Half is:"+str(righthalf))
		merge_sort(lefthalf)
		merge_sort(righthalf)

		#Merging Step
		i = 0
		j = 0 
		k = 0 #Final Sorted Array counter

		while i<len(lefthalf) and j<len(righthalf):
			#Check which one is greater
			if lefthalf[i] < righthalf[j]:
				arr[k] = lefthalf[i]
				i+=1
			else:
				arr[k] = righthalf[j]
				j+=1

			k+=1

		#Now i or j might not have reached the end

		while i<len(lefthalf):
			arr[k] = arr[i]
			i+=1
			k+=1
		while j<len(righthalf):
			arr[k] = arr[j]
			j+=1
			k+=1
	return arr

print(merge_sort([5,6,1,2,9,10]))