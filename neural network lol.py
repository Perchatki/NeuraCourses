import numpy
import scipy.special
import matplotlib.pyplot

class Neuralnetwork:
    def __init__(self, innode, hidnode, outnode, learnrate):
        
        self.inp_n=innode #input nodes
        self.hid_n=hidnode #hidden nodes
        self.out_n=outnode #output nodes
        self.l_r=learnrate
        
        self.activ_func = lambda x: scipy.special.expit(x)
        
        #генерация матриц слоев
        #wih = weights input-hidden
        self.wih = (numpy.random.normal(0, pow(self.hid_n, -0.5),(self.hid_n, self.inp_n)) )
        #who = weights hidden-output
        self.who = (numpy.random.normal(0, pow(self.hid_n, -0.5),(self.out_n, self.hid_n)) )
        
    #опрос сети
    def query(self, input_l):
        
        #normalisation(sigmoid)
        
        inputs = numpy.array(input_l, ndmin=2).T
        
        #hidden hid_i = hidden_inputs hid_o = hidden_outputs
        hid_i = numpy.dot(self.wih, inputs)
        hid_o = self.activ_func(hid_i)
        
        #outputs out_i = final_inputs out_o = final_outputs
        out_i = numpy.dot(self.who, hid_o)
        out_o = self.activ_func(out_i)
        
        return out_o
    
        pass
        
    def train(self, input_l, target_l):
        
        inputs = numpy.array(input_l, ndmin=2).T
        #targets - Целевое значение
        targets = numpy.array(target_l, ndmin=2).T
        
        
        
        #hidden hid_i = hidden_inputs hid_o = hidden_outputs
        self.hid_i = numpy.dot(self.wih,inputs)
        hid_o=self.activ_func(self.hid_i)
        
        #outputs out_i = final_inputs out_o = final_outputs
        self.out_i = numpy.dot(self.who, hid_o)
        out_o=self.activ_func(self.out_i)
        
        out_e = targets - out_o
        hid_e = numpy.dot(self.who.T,out_e)
        
        #обновление слоев
        self.who += self.l_r * numpy.dot((out_e * out_o * (1-out_o)), numpy.transpose(hid_o))
        self.wih += self.l_r * numpy.dot((hid_e * hid_o * (1-hid_o)), numpy.transpose(inputs))
        pass
inpn = 784
hidn = 100
outn = 10
lr   = 0.3

nn1 = Neuralnetwork(inpn, hidn, outn, lr)

data_file=open("C:/Users/Danii/Desktop/progs/programming/Jupiter/mnist_train_100.csv",'r')
data_list=data_file.readlines()
data_file.close()

for record in data_list[0:100]:
    values = record.split(',')
    inputss = numpy.asfarray(values[1:])
    targetss = numpy.zeros(outn) + 0.01
    targetss[int(values[0])] = 0.99
    nn1.train(inputss,targetss)


num = 0
values = data_list[num].split(',')
colarray = numpy.asfarray(values[1:])
image = colarray.reshape(28, 28)
matplotlib.pyplot.imshow(image, cmap = 'Greys', interpolation = 'None')
print(data_list[num][0])
