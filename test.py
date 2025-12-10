width = 12
stage = 4

for i in range(0, stage):
    splits = 2 ** i
    for j in range(0, splits):
        a_start = width-1 - (j*(2*(width//(2*(i+1)))))
        a_end = width - (j*(2*(width//(2*(i+1))))) - (width//(2*(i+1)))
        b_start = width - 1 - (j*(2*(width//(2*(i+1))))) - (width//(2*(i+1)))
        b_end = width - (j*(2*(width//(2*(i+1))))) - 2*(width//(2*(i+1)))
        print(f"i: {i} j: {j} a[{a_start}:{a_end}] b[{b_start}:{b_end}]")
    print()