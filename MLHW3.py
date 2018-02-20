import math
import numpy


#get all data and separate them into diff array

filein = open("Crime-Training.txt","r").read().split("\n")
eachinput = [temp.split("\t")[:96] for temp in filein]
Training = numpy.array(eachinput, dtype=float)
eachinput2 = [temp.split("\t")[1:96] for temp in filein]
Training3 = numpy.array(eachinput2, dtype=float)
eachinput3 = [temp.split("\t")[0] for temp in filein]
Training4 = numpy.array(eachinput3, dtype=float)
Training2 = numpy.concatenate((Training[0:,1:],numpy.ones((1595,1), dtype=numpy.int)),axis=1)

filein = open("Crime-Test.txt","r").read().split("\n")
eachinput = [temp.split("\t")[:96] for temp in filein]
Testing = numpy.array(eachinput, dtype=float)
Testing2 = numpy.concatenate((Testing[0:,1:],numpy.ones((399,1), dtype=numpy.int)),axis=1)

#print each item
#for item in Training:
# print(item)

# w= (x^Tx)^-1X^TY
#x^Tx
InsInv = numpy.dot(Training2.transpose(),Training2)
#(x^Tx)^-1
Inv = numpy.linalg.inv(InsInv)
#X^TY
Last = numpy.dot(Training2.transpose(),Training[:,0])
#(x^Tx)^-1X^TY
w=numpy.dot(Inv,Last)
#print w
yNew = numpy.dot(Training2,w)
#RMSE = root(sum(ynew-y)^2/1595)
MSE = numpy.sum((yNew-Training[:,0])**2)
RMSE =math.sqrt(MSE/1595)
print "!!!!!!!!Q1 Linear Regression for Closed form!!!!!!!!!!!!"
print "LR Training MRSE: ",
print RMSE

yNew = numpy.dot(Testing2,w)
#RMSE = root(sum(ynew-y)^2/1595)
MSE = numpy.sum((yNew-Testing[:,0])**2)
RMSE =math.sqrt(MSE/399)
print "LR Test MRSE: ",
print RMSE
print "!!!!!!!!!!!!!!!!!!!!"


print "!!!!!!!!Q2 Ridge Regression for Closed Form!!!!!!!!!!!!"
#RR
#1595/5=319
block1=Training2[0:319]
block2=Training2[319:638]
block3=Training2[638:957]
block4=Training2[957:1276]
block5=Training2[1276:1595]

Yblock1=Training[0:319,0]
Yblock2=Training[319:638,0]
Yblock3=Training[638:957,0]
Yblock4=Training[957:1276,0]
Yblock5=Training[1276:1595,0]

Lamda = 400.0
#1      2       3       4   5       6       7       8       9           10
#400 -> 200 -> 100 -> 50 -> 25 -> 12.5 -> 6.25 -> 3.125 -> 1.5625 -> 0.78125
while (Lamda >= .78):
    total=0.0
    for x in range(5):
        if (x==0):
            block=numpy.concatenate([block1, block2, block3, block4])
            Yblock=numpy.concatenate([Yblock1, Yblock2, Yblock3, Yblock4])
            Testblock=block5
            TestYblock=Yblock5
        elif (x==1):
            block=numpy.concatenate([block1, block2, block3, block5])
            Yblock=numpy.concatenate([Yblock1, Yblock2, Yblock3, Yblock5])
            Testblock=block4
            TestYblock=Yblock4
        elif (x==2):
            block=numpy.concatenate([block1, block2, block4, block5])
            Yblock=numpy.concatenate([Yblock1, Yblock2, Yblock4, Yblock5])
            Testblock=block3
            TestYblock=Yblock3
        elif (x==3):
            block=numpy.concatenate([block1, block3, block4, block5])
            Yblock=numpy.concatenate([Yblock1, Yblock3, Yblock4, Yblock5])
            Testblock=block2
            TestYblock=Yblock2
        else:
            block=numpy.concatenate([block2, block3, block4, block5])
            Yblock=numpy.concatenate([Yblock2, Yblock3, Yblock4, Yblock5])
            Testblock=block1
            TestYblock=Yblock1
        #w =(X^TX+LI)^-1X^TY
        Inv=numpy.linalg.inv(numpy.dot(block.transpose(),block) + Lamda*numpy.identity(96))
        Last=numpy.dot(block.transpose(),Yblock)
        w=numpy.dot(Inv,Last)
        yNew=numpy.dot(Testblock,w)
        MSE=numpy.sum((yNew-TestYblock)**2)
        RMSE=math.sqrt(MSE/319)
        total += RMSE
    print "================="
    print "Lamda: ",
    print Lamda,
    print " Avg: ",
    print total/5
    #Print actual vals with lamda
    InsInv = numpy.dot(Training2.transpose(),Training2) + Lamda*numpy.identity(96)
    #(x^Tx)^-1
    Inv = numpy.linalg.inv(InsInv)
    #X^TY
    Last = numpy.dot(Training2.transpose(),Training[:,0])
    #(x^Tx)^-1X^TY
    w=numpy.dot(Inv,Last)

    yNew = numpy.dot(Training2,w)
    #RMSE = root(sum(ynew-y)^2/1595)
    MSE = numpy.sum((yNew-Training[:,0])**2)
    RMSE =math.sqrt(MSE/1595)
    print "LR Training MRSE: ",
    print RMSE

    yNew = numpy.dot(Testing2,w)
    #RMSE = root(sum(ynew-y)^2/1595)
    MSE = numpy.sum((yNew-Testing[:,0])**2)
    RMSE =math.sqrt(MSE/399)
    print "LR Test MRSE: ",
    print RMSE

    Lamda = Lamda/2
print "!!!!!!!!!!!!!!!!!!!!"


########################################################################
# #Test Section
# test = numpy.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
#
# #   1   2   3
# #   4   5   6
# #   7   8   9
# #[index for start row: ,]
# print test[0]
# print test[:1,]
# print test[:1,0]
# print test[1:,0]
# print test[1:2,0]
# print "Swap : side"
# print test[1,1:]
# print test[0,:]
# print test[0,1:]
# print test[1,1:]
# print test[1,1:]
# print "Darray"
# print test[1:,1:]
# print test[0:,1:]
########################################################################
#Shits not working. Redoing from start for part 3 and 4....

#reading everything again from scaratch cuz top is now too complicated and I might have changed the matrix
filein = open("Crime-Training.txt","r").read().split("\n")
Training = []
YTraining = []
for temp in filein:
    temp = temp.split("\t")
    temp = [float(i) for i in temp]
    Training.append(temp[1:] + [1])
    YTraining.append([temp[0]])
YTraining = numpy.matrix(YTraining)
Training = numpy.matrix(Training)

filein = open("Crime-Test.txt","r").read().split("\n")
Test = []
YTest = []
filein = [eachline.rstrip('\n') for eachline in filein]
for temp in filein:
    temp = temp.split("\t")
    temp = [float(i) for i in temp]
    Test.append(temp[1:] + [1])
    YTest.append([temp[0]])
Test = numpy.matrix(Test)
YTest = numpy.matrix(YTest)


print "!!!!!!!!Q3 Linear Regression with Gadient Descent!!!!!!!!!!!!"
#Gradient Descent
transpose=Training2.transpose()
w = numpy.random.normal(0, 1, size=(96, 1))
delta=10e-6
j=0
dif = 0

while True:
    wNew = w + delta * transpose * (YTraining - Training * w)
    dif = numpy.linalg.norm(wNew - w)
    w = wNew
    if dif < 10 ** -7:
        break

#Training
tot = 0
for i, test in enumerate(Training):
    yNew = w.transpose() * test.transpose()
    tot += (YTraining[i].item(0) - yNew.item(0)) ** 2
RMSE = numpy.sqrt(tot / 1595)
print "LR Gradient Descent for Training:",
print RMSE

#Test
tot = 0
for i, test in enumerate(Test):
    yNew = w.transpose() * test.transpose()
    tot += (YTest[i].item(0) - yNew.item(0)) ** 2
RMSE = numpy.sqrt(tot / 399)
print "LR Gradient Descent for Test:",
print RMSE

print "!!!!!!!!!!!!!!!!!!!!"

print "!!!!!!!!Q4 Ridge Regression with Gadient Descent!!!!!!!!!!!!"
#Gradient Descent #Try out with 25
w = numpy.random.normal(0, 1, size=(96, 1))
delta=10e-6
j=0
dif = 0

while True:
    wNew = w + delta * (transpose * (YTraining - Training * w) - 25*w)
    dif = numpy.linalg.norm(wNew - w)
    w = wNew
    if dif < 10 ** -7:
        break

#Training
tot = 0
for i, test in enumerate(Training):
    yNew = w.transpose() * test.transpose()
    tot += (YTraining[i].item(0) - yNew.item(0)) ** 2
RMSE = numpy.sqrt(tot / 1595)
print "RR Gradient Descent for Training:",
print RMSE

#Test
tot = 0
for i, test in enumerate(Test):
    yNew = w.transpose() * test.transpose()
    tot += (YTest[i].item(0) - yNew.item(0)) ** 2
RMSE = numpy.sqrt(tot / 399)
print "LR Gradient Descent for Test:",
print RMSE

print "!!!!!!!!!!!!!!!!!!!!"
