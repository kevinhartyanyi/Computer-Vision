with open("image1.txt", 'rU') as reader:
    text = reader.read()

lines = text.splitlines()
first_line = []
second_line = []
third_line = []
fourth_line = []
print(lines)
for l in lines[:10]:
    l = [e for e in l.split(' ') if e != '']
    print(l)
    fi,s,t,fo = l
    first_line.append(fi)
    second_line.append(s)
    third_line.append(t)
    fourth_line.append(fo)

print(first_line)
print(' '.join(list(first_line)))

print(len(first_line))
with open("res.txt", "w") as cout:
    cout.write(' '.join(list(first_line)) + '\n')
    cout.write(' '.join(list(second_line)) + '\n')
    cout.write(' '.join(list(third_line)) + '\n')
    cout.write(' '.join(list(fourth_line)) + '\n')