outf = open("offsets.txt", 'w')
offset = 0
offsets = []

with open("/mnt/d/vocsamp.h2.y") as file:
    for line in file:
        outf.write("{}\n".format(offset))
        offset += len(line)

outf.close()
