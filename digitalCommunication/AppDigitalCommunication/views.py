from django.shortcuts import render
from django.http import JsonResponse
import json
import numpy as np
from ModulationPy import PSKModem
import numpy as np
from ModulationPy import QAMModem
from scipy import special
import matplotlib.pyplot as plt
from io import BytesIO
from .utils import graph


# Create your views here.

 
encoded_output = 0
 
def index(request):
    return render(request, 'AppDigitalCommunication/index.html')







def source_coding(request):
    data = request.POST.get('inpAdd')
    
    class Nodes:
        def __init__(self, probability, symbol, left=None, right=None):
            # probability of the symbol  
            self.probability = probability

            # the symbol  
            self.symbol = symbol

            # the left node  
            self.left = left

            # the right node  
            self.right = right

            # the tree direction (0 or 1)  
            self.code = ''


    """ A supporting function in order to calculate the probabilities of symbols in specified data """


    def calculateProbability(the_data):
        the_symbols = dict()
        for item in the_data:
            if the_symbols.get(item) == None:
                the_symbols[item] = 1
            else:
                the_symbols[item] += 1
        return the_symbols


    """ A supporting function in order to print the codes of symbols by travelling a Huffman Tree """
    the_codes = dict()


    def CalculateCodes(node, value=''):
        # a huffman code for current node  
        newValue = value + str(node.code)

        if (node.left):
            CalculateCodes(node.left, newValue)
        if (node.right):
            CalculateCodes(node.right, newValue)

        if (not node.left and not node.right):
            the_codes[node.symbol] = newValue

        return the_codes


    """ A supporting function in order to get the encoded result """


    def OutputEncoded(the_data, coding):
        encodingOutput = []
        for element in the_data:
            # print(coding[element], end = '')  
            encodingOutput.append(coding[element])

        the_string = ''.join([str(item) for item in encodingOutput])
        return the_string


    """ A supporting function in order to calculate the space difference between compressed and non compressed data"""


    def TotalGain(the_data, coding):
        # total bit space to store the data before compression  
        beforeCompression = len(the_data) * 8
        afterCompression = 0
        the_symbols = coding.keys()
        for symbol in the_symbols:
            the_count = the_data.count(symbol)
            # calculating how many bit is required for that symbol in total  
            afterCompression += the_count * len(coding[symbol])
        # print("Space usage before compression (in bits):", beforeCompression)
        # print("Space usage after compression (in bits):", afterCompression)
        return beforeCompression, afterCompression


    def HuffmanEncoding(the_data):
        symbolWithProbs = calculateProbability(the_data)
        the_symbols = symbolWithProbs.keys()
        the_probabilities = symbolWithProbs.values()
        # print("symbols: ", the_symbols)
        # print("probabilities: ", the_probabilities)

        the_nodes = []

        # converting symbols and probabilities into huffman tree nodes  
        for symbol in the_symbols:
            the_nodes.append(Nodes(symbolWithProbs.get(symbol), symbol))

        while len(the_nodes) > 1:
            # sorting all the nodes in ascending order based on their probability  
            the_nodes = sorted(the_nodes, key=lambda x: x.probability)
            # for node in nodes:    
            #      print(node.symbol, node.prob)  

            # picking two smallest nodes  
            right = the_nodes[0]
            left = the_nodes[1]

            left.code = 0
            right.code = 1

            # combining the 2 smallest nodes to create new node  
            newNode = Nodes(left.probability + right.probability, left.symbol + right.symbol, left, right)

            the_nodes.remove(left)
            the_nodes.remove(right)
            the_nodes.append(newNode)

        huffmanEncoding = CalculateCodes(the_nodes[0])
        # print("symbols with codes", huffmanEncoding)
        TotalGain(the_data, huffmanEncoding)
        encodedOutput = OutputEncoded(the_data, huffmanEncoding)
        return encodedOutput, the_nodes[0], the_symbols, the_probabilities, symbolWithProbs, huffmanEncoding
    


    def HuffmanDecoding(encodedData, huffmanTree):
        treeHead = huffmanTree
        decodedOutput = []
        for x in encodedData:
            if x == '1':
                huffmanTree = huffmanTree.right
            elif x == '0':
                huffmanTree = huffmanTree.left
            try:
                if huffmanTree.left.symbol == None and huffmanTree.right.symbol == None:
                    pass
            except AttributeError:
                decodedOutput.append(huffmanTree.symbol)
                huffmanTree = treeHead

        string = ''.join([str(item) for item in decodedOutput])
        return string

    the_data = str(data)
    #print(the_data)
    
    encoding, the_tree, the_symbols, the_probabilities, symbolWithProbs, huffmanEncoding   = HuffmanEncoding(the_data)
    encoded_output = encoding
    
    huffmanDecoding = HuffmanDecoding(encoded_output, the_tree)
    # beforeCompression, afterCompression = TotalGain(the_data, coding)
    
    beforeCompression, afterCompression = TotalGain(the_data, huffmanEncoding)


# GLOBAL VARIABLE
    # request.session['encoded_output'] = encoded_output
    # print("symbols: ", the_symbols)
    # print("probabilities: ", the_probabilities)
    # print("symbolWithProbs ", symbolWithProbs)
    # print("Symbols with code",huffmanEncoding)
    # print("Space usage before compression (in bits):", beforeCompression)
    # print("Space usage after compression (in bits):", afterCompression)
    # print("Space usage before compression (in bits):", beforeCompression)
    # print("Space usage after compression (in bits):", afterCompression)
 
    # decoded_Output=HuffmanDecoding(encoding, the_tree)
    

    
    return render(request, 'AppDigitalCommunication/source_coding.html', {'data':data, 'encoded_output':encoded_output, "symbols": the_symbols, "probabilities":the_probabilities, "symbolWithProbs":symbolWithProbs, "Symbols_with_code":huffmanEncoding, 'beforeCompression':beforeCompression, 'afterCompression':afterCompression,'huffmanDecoding':huffmanDecoding})
 
 
 
 

def channel_coding(request):
    data = request.POST.get('inpAdd1')
    
    class Convolutional:
        def moveOnMachine(self, state, input):
            nextState = 0
            output = ""
            if (state == 0 and input == "0"):
                nextState = 0
                output = "00"
            elif (state == 0 and input == "1"):
                nextState = 2
                output = "11"
            elif (state == 1 and input == "0"):
                nextState = 0
                output = "10"
            elif (state == 1 and input == "1"):
                nextState = 2
                output = "01"
            elif (state == 2 and input == "0"):
                nextState = 1
                output = "11"
            elif (state == 2 and input == "1"):
                nextState = 3
                output = "00"
            elif (state == 3 and input == "0"):
                nextState = 1
                output = "01"
            else:
                nextState = 3
                output = "10"
            return (nextState, output)

        def encode(self, input):
            i = 0
            state = 0
            ans = ''
            while (i < len(input)):
                state, output = self.moveOnMachine(state, input[i])
                ans = ans + output
                i = i + 1
            return ans

        def constructPath(self, path, index):
            code = ""
            length = len(path[0])
            thisState = index
            for i in range(length - 1, -1, -1):
                if (thisState == 0):
                    code = code + "0"
                elif (thisState == 1):
                    code = code + "0"
                elif (thisState == 2):
                    code = code + "1"
                else:
                    code = code + "1"
                if (i == -1):
                    thisState = 0
                else:
                    thisState = path[thisState][i]

            return (code[::-1])

        def decode(self, input):
            currentPM = [None] * 4
            nextPM = [None] * 4
            path = [[0 for x in range(len(input) / 2)] for y in range(4)]
            currentPM[0] = 0
            currentPM[1] = float("inf")
            currentPM[2] = float("inf")
            currentPM[3] = float("inf")

            i = 0
            while (i < len(input)):
                str = input[i: i + 2]
                if (str == '00'):
                    if (currentPM[0] < currentPM[1] + 1):
                        nextPM[0] = currentPM[0]
                        path[0][i / 2] = 0
                    else:
                        nextPM[0] = currentPM[1] + 1
                        path[0][i / 2] = 1

                    if (currentPM[2] + 2 < currentPM[3] + 1):
                        nextPM[1] = currentPM[2] + 2
                        path[1][i / 2] = 2
                    else:
                        nextPM[1] = currentPM[3] + 1
                        path[1][i / 2] = 3

                    if (currentPM[0] + 2 < currentPM[1] + 1):
                        nextPM[2] = currentPM[0] + 2
                        path[2][i / 2] = 0
                    else:
                        nextPM[2] = currentPM[1] + 1
                        path[2][i / 2] = 1

                    if (currentPM[2] < currentPM[3] + 1):
                        nextPM[3] = currentPM[2]
                        path[3][i / 2] = 2
                    else:
                        nextPM[3] = currentPM[3] + 1
                        path[3][i / 2] = 3
                ###############################

                elif (str == '01'):
                    if (currentPM[0] + 1 < currentPM[1] + 2):
                        nextPM[0] = currentPM[0] + 1
                        path[0][i / 2] = 0
                    else:
                        nextPM[0] = currentPM[1] + 2
                        path[0][i / 2] = 1

                    if (currentPM[2] + 1 < currentPM[3]):
                        nextPM[1] = currentPM[2] + 1
                        path[1][i / 2] = 2
                    else:
                        nextPM[1] = currentPM[3]
                        path[1][i / 2] = 3

                    if (currentPM[0] + 1 < currentPM[1]):
                        nextPM[2] = currentPM[0] + 1
                        path[2][i / 2] = 0
                    else:
                        nextPM[2] = currentPM[1]
                        path[2][i / 2] = 1

                    if (currentPM[2] + 1 < currentPM[3] + 2):
                        nextPM[3] = currentPM[2] + 1
                        path[3][i / 2] = 2
                    else:
                        nextPM[3] = currentPM[3] + 2
                        path[3][i / 2] = 3
                ###############################    

                elif (str == '10'):
                    if (currentPM[0] + 1 < currentPM[1]):
                        nextPM[0] = currentPM[0] + 1
                        path[0][i / 2] = 0
                    else:
                        nextPM[0] = currentPM[1]
                        path[0][i / 2] = 1

                    if (currentPM[2] + 1 < currentPM[3] + 2):
                        nextPM[1] = currentPM[2] + 1
                        path[1][i / 2] = 2
                    else:
                        nextPM[1] = currentPM[3] + 2
                        path[1][i / 2] = 3

                    if (currentPM[0] + 1 < currentPM[1] + 2):
                        nextPM[2] = currentPM[0] + 1
                        path[2][i / 2] = 0
                    else:
                        nextPM[2] = currentPM[1] + 2
                        path[2][i / 2] = 1

                    if (currentPM[2] + 1 < currentPM[3]):
                        nextPM[3] = currentPM[2] + 1
                        path[3][i / 2] = 2
                    else:
                        nextPM[3] = currentPM[3]
                        path[3][i / 2] = 3
                #########################################
                elif (str == "11"):
                    if (currentPM[0] + 2 < currentPM[1] + 1):
                        nextPM[0] = currentPM[0] + 2
                        path[0][i / 2] = 0
                    else:
                        nextPM[0] = currentPM[1] + 1
                        path[0][i / 2] = 1

                    if (currentPM[2] < currentPM[3] + 1):
                        nextPM[1] = currentPM[2]
                        path[1][i / 2] = 2
                    else:
                        nextPM[1] = currentPM[3] + 1
                        path[1][i / 2] = 3

                    if (currentPM[0] < currentPM[1] + 1):
                        nextPM[2] = currentPM[0]
                        path[2][i / 2] = 0
                    else:
                        nextPM[2] = currentPM[1] + 1
                        path[2][i / 2] = 1

                    if (currentPM[2] + 2 < currentPM[3] + 1):
                        nextPM[3] = currentPM[2] + 2
                        path[3][i / 2] = 2
                    else:
                        nextPM[3] = currentPM[3] + 1
                        path[3][i / 2] = 3

                i = i + 2
                currentPM = nextPM[:]

            index = currentPM.index(min(currentPM))
            print('min error = ', min(currentPM))
            return self.constructPath(path, index)


    def main():
        # encoded_output = request.session['encoded_output'] 
       
        c = Convolutional();
        raw_input = str(data)
        code = c.encode(raw_input)
        print('code= ', code)
        # decoded = c.decode(code)
        # print('decode = ', decoded)
        return code, raw_input

    
    # if __name__ == "__main__":
    convCod, raw_input = main()
    return render(request, 'AppDigitalCommunication/channel_coding.html', {'convCod':convCod, 'decoded':raw_input})





def modulation(request):
    data = request.POST.get('inpAdd1')
    string = str(data)
    if string=='None':
        to_list = [0, 1, 1, 0]
    else:
        to_list = list(string)
        to_list = [int(i) for i in to_list]
    
    #qpsk
    modem = PSKModem(4, np.pi/4, 
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

    msg = np.array(to_list) # input message

    pskmodulated = modem.modulate(msg) # modulation
    pskdemodulated = modem.demodulate(pskmodulated) # demodulation

    # print("Modulated message:\n"+str(pskmodulated))
    # print("Demodulated message:\n"+str(pskdemodulated))
    
    
    
    # modem = PSKModem(8, np.pi/4, 
    #              bin_input=True,
    #              soft_decision=False,
    #              bin_output=True)

    # # msg = np.array([0, 1, 1, 0]) # input message

    # fourqammodulated = modem.modulate(msg) # modulation
    # qamdemodulated = modem.demodulate(fourqammodulated) # demodulation 
     
    #16-QAM
    
    modem = PSKModem(16, np.pi/4, 
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

    # msg = np.array([0, 1, 1, 0]) # input message

    qammodulated = modem.modulate(msg) # modulation
    qamDemodulated = modem.demodulate(qammodulated) # demodulation  

    return render(request, 'AppDigitalCommunication/modulation.html', {'pskmodulated':pskmodulated,'pskdemodulated':pskdemodulated, "qammodulated":qammodulated, 'qamDemodulated':qamDemodulated})
    
    






def channel(request):
    chart = graph()
    return render(request,  'AppDigitalCommunication/modulation.html', {'chart':chart})
