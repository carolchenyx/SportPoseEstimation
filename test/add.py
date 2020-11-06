from random import randint
f=open('game.txt')
score=f.read().split()
game_times=int(score[0])
min_times=int(score[1])
total_times=int(score[2])
if game_times!=0:
    avg_times=float(total_times)/game_times
else:
    avg_times=0
print('你已经玩了%d次，最少%d次猜出结果，平均%.2f轮猜出结果'%(game_times,min_times,avg_times))
num=randint(1,100)
times=0
print('Guess what I think?')
Bingo=False

while Bingo==False:
    answer = int(input())
    if answer>num:
        print('too big!')
    if answer<num:
        print('too small!')
    if answer==num:
        print('BINGO!')
        Bingo=True
if game_times==0 or times<min_times:
    min_times=times
total_times+=times
game_times+=1
result='%d%d%d'%(game_times,min_times,total_times)
f=open('game.txt','w')
f.write(result)
f.close()
