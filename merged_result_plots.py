# imports
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# unpickle
with open("/home/tmomose/SuttonRoomData/20171102/20171102-1748_training-history(plan2).pkl","rb") as f:
    ph = pkl.load(f)
with open("/home/tmomose/SuttonRoomData/20171102/20171102-1649_training-history(options2).pkl","rb") as f:
    sh = pkl.load(f)
with open("/home/tmomose/SuttonRoomData/20171102/20171102-1326_training-history(flat).pkl","rb") as f:
    qh_raw = pkl.load(f)
# older flat q histories might not have the choices line and need reordering
qh = np.zeros(ph.shape)
if qh_raw.shape[1] == 6:
    qh[:,:4] = qh_raw[:,:4]
    qh[:,4]  = qh_raw[:,5]
    qh[:,5]  = qh_raw[:,4]
    qh[:,6]  = qh_raw[:,4] # steps and choices are same for flat q

# x-axis array setup
e = np.arange(qh.shape[0])    # episode x-axis
d = np.arange(qh.shape[0]-10)+5 # episode x-axis for 10-episode averages

# compute 10-episode averages
pd = np.zeros((len(d),7))
sd = np.zeros((len(d),7))
qd = np.zeros((len(d),7))
for i in range(len(d)):
    pd[i,:] = np.mean(ph[i:i+10,:],axis=0)
    sd[i,:] = np.mean(sh[i:i+10,:],axis=0)
    qd[i,:] = np.mean(qh[i:i+10,:],axis=0)

# prepare lists for plotting
he=[qh,sh,ph]
hd=[qd,sd,pd]
ce=["lightblue","palegreen","lightpink"]
cd=["b","g","r"]
lab=["Flat RL","Standard HRL","Planning HRL"]

## Contents of hist pickle as of 2017/11/2:
##  0 training step
##  1 avg_td
##  2 avg_ret
##  3 avg_test_ret
##  4 avg_test_successrate
##  5 avg_test_steps
##  6 avg_test_choices
## (7 avg td for LLC policy)
STP0=0
TDE0=1
RET0=2
RET1=3
SUC1=4
STP1=5
CHO1=6

# plot steps per test episode vs. training episode
print("Plotting steps per test episode vs. training episode")
for i in range(3):
    plt.loglog(e,he[i][:,STP1],c=ce[i],basex=10,basey=10)
for i in range(3):
    plt.loglog(d,hd[i][:,STP1],c=cd[i],label=lab[i],basex=10,basey=10)
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Episodes",fontsize="x-large")
plt.ylabel("Steps per Test Episode",fontsize="x-large")
plt.show()

# plot choices per test episode vs. training episode
print("Plotting choices per test episode vs. training episode")
for i in range(3):
    plt.loglog(e,he[i][:,CHO1],c=ce[i],basex=10,basey=10)
for i in range(3):
    plt.loglog(d,hd[i][:,CHO1],c=cd[i],label=lab[i],basex=10,basey=10)
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Episodes",fontsize="x-large")
plt.ylabel("Choices per Test Episode",fontsize="x-large")
plt.show()

# plot average test return (discounted) vs. training episode
print("Plotting average test return vs. training episode")
for i in range(3):
    plt.semilogx(e,he[i][:,RET1],c=ce[i])
for i in range(3):
    plt.semilogx(d,hd[i][:,RET1],c=cd[i],label=lab[i])
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Episodes",fontsize="x-large")
plt.ylabel("Average Test Return",fontsize="x-large")
plt.show()

# plot success rate vs. training episode
print("Plotting average test success rate vs. training episode")
for i in range(3):
    plt.semilogx(e,he[i][:,SUC1],c=ce[i])
for i in range(3):
    plt.semilogx(d,hd[i][:,SUC1],c=cd[i],label=lab[i])
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Episodes",fontsize="x-large")
plt.ylabel("Test Success Rate",fontsize="x-large")
plt.show()

# plot steps per test episode vs. training steps
print("Plotting steps per test episode vs. training steps")
for i in range(3):
    plt.loglog(he[i][:,STP0],he[i][:,STP1],c=ce[i],basex=10,basey=10)
for i in range(3):
    plt.loglog(hd[i][:,STP0],hd[i][:,STP1],c=cd[i],label=lab[i],basex=10,basey=10)
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Steps",fontsize="x-large")
plt.ylabel("Steps per Test Episode",fontsize="x-large")
plt.show()

# plot choices per test episode vs. training steps
print("Plotting choices per test episode vs. training steps")
for i in range(3):
    plt.loglog(he[i][:,STP0],he[i][:,CHO1],c=ce[i],basex=10,basey=10)
for i in range(3):
    plt.loglog(hd[i][:,STP0],hd[i][:,CHO1],c=cd[i],label=lab[i],basex=10,basey=10)
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Steps",fontsize="x-large")
plt.ylabel("Choices per Test Episode",fontsize="x-large")
plt.show()

# plot average test return (discounted) vs. training steps
print("Plotting average test return vs. training steps")
for i in range(3):
    plt.semilogx(he[i][:,STP0],he[i][:,RET1],c=ce[i])
for i in range(3):
    plt.semilogx(hd[i][:,STP0],hd[i][:,RET1],c=cd[i],label=lab[i])
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Steps",fontsize="x-large")
plt.ylabel("Average Test Return",fontsize="x-large")
plt.show()

# plot success rate vs. training steps
print("Plotting average test success rate vs. training steps")
for i in range(3):
    plt.semilogx(he[i][:,STP0],he[i][:,SUC1],c=ce[i])
for i in range(3):
    plt.semilogx(hd[i][:,STP0],hd[i][:,SUC1],c=cd[i],label=lab[i])
plt.grid()
plt.legend(fontsize="large")
plt.tick_params(axis="both",which="major",labelsize="large")
plt.xlabel("Training Steps",fontsize="x-large")
plt.ylabel("Test Success Rate",fontsize="x-large")
plt.show()

