# 1* READING & WRITING FILES
# 1.8 Read TXT Files
```py
f = open("files/Python.txt", "r") #opens file with name of "Python.txt"
print(f.read()) # read and print the entire file
f.close() # remember to colse the file

# Life is short,
# Use Python!
```
```py
f = open("files/Python.txt", "r") #opens file with name of "Python.txt"
print(f.readline()) # read the 1st line
print(f.readline()) # read the next line
f.close()

# Life is short,
#
# Use Python!
```
```py
f = open("files/Python.txt", "r") 
myList = []
for line in f:
    myList.append(line)
f.close()
print(myList)

# ['Life is short,\n', 'Use Python!']
```
# 1.9 Write TXT Files
```py
# Write file with name of "test.txt"
f = open("files/test.txt","w")  
f.write("I love Python.\n")
f.write("I will be a Python master.\n")
f.write("I need to keep learning!")
f.close()

# read and see the test.txt file
f = open("files/test.txt","r") 
print(f.read())
f.close()

# I love Python.
# I will be a Python master.
# I need to keep learning!
```
# 1.10 Read CSV Files
```py
```
