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
    


    # def HuffmanDecoding(encodedData, huffmanTree):
    #     treeHead = huffmanTree
    #     decodedOutput = []
    #     for x in encodedData:
    #         if x == '1':
    #             huffmanTree = huffmanTree.right
    #         elif x == '0':
    #             huffmanTree = huffmanTree.left
    #         try:
    #             if huffmanTree.left.symbol == None and huffmanTree.right.symbol == None:
    #                 pass
    #         except AttributeError:
    #             decodedOutput.append(huffmanTree.symbol)
    #             huffmanTree = treeHead

    #     string = ''.join([str(item) for item in decodedOutput])
    #     return string

    the_data = str(data)
    #print(the_data)
    global encoded_output 
    encoding, the_tree, the_symbols, the_probabilities, symbolWithProbs, huffmanEncoding   = HuffmanEncoding(the_data)
    encoded_output = encoding
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
    

    
    return render(request, 'AppDigitalCommunication/source_coding.html', {'data':data, 'encoded_output':encoded_output, "symbols": the_symbols, "probabilities":the_probabilities, "symbolWithProbs":symbolWithProbs, "Symbols_with_code":huffmanEncoding, 'beforeCompression':beforeCompression, 'afterCompression':afterCompression})
 
 
 
 

def channel_coding(request):

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
        raw_input = encoded_output
        code = c.encode(raw_input)
        print('code= ', code)
        # decoded = c.decode(code)
        # print('decode = ', decoded)
        return code

    global convCod 
    # if __name__ == "__main__":
    convCod = main()
    return render(request, 'AppDigitalCommunication/channel_coding.html', {'convCod':convCod})

def modulation(request):
    #qpsk
    modem = PSKModem(4, np.pi/4, 
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

    msg = np.array(convCod) # input message

    pskmodulated = modem.modulate(msg) # modulation
    demodulated = modem.demodulate(pskmodulated) # demodulation

    print("Modulated message:\n"+str(pskmodulated))
    print("Demodulated message:\n"+str(demodulated))
    
     
    #16-QAM
    
    modem = PSKModem(16, np.pi/4, 
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

    msg = np.array(convCod) # input message

    qammodulated = modem.modulate(msg) # modulation
    demodulated = modem.demodulate(qammodulated) # demodulation  

    return render(request, 'AppDigitalCommunication/modulation.html', {'pskmodulated':pskmodulated, "qammodulated":qammodulated})
    
    






def channel(request):
    def BER_calc(a, b):
        num_ber = np.sum(np.abs(a - b))
        ber = np.mean(np.abs(a - b))
        return int(num_ber), ber


    def BER_qam(M, EbNo):
        EbNo_lin = 10 ** (EbNo / 10)
        if M > 4:
            P = 2 * np.sqrt((np.sqrt(M) - 1) /
                            (np.sqrt(M) * np.log2(M))) * special.erfc(np.sqrt(EbNo_lin * 3 * np.log2(M) / 2 * (M - 1)))
        else:
            P = 0.5 * special.erfc(np.sqrt(EbNo_lin))
        return P


    EbNos = np.array([i for i in range(30)])  # array of Eb/No in dBs
    N = 100000  # number of symbols per the frame
    N_c = 100  # number of trials

    Ms = [4, 16, 64, 256]  # modulation orders

    ''' Simulation loops '''

    mean_BER = np.empty((len(EbNos), len(Ms)))
    for idxM, M in enumerate(Ms):
        print("Modulation order: ", M)
        BER = np.empty((N_c,))
        k = np.log2(M)  # number of bit per modulation symbol

        modem = QAMModem(M,
                        bin_input=True,
                        soft_decision=False,
                        bin_output=True)

        for idxEbNo, EbNo in enumerate(EbNos):
            print("EbNo: ", EbNo)
            snrdB = EbNo + 10 * np.log10(k)  # Signal-to-Noise ratio (in dB)
            noiseVar = 10 ** (-snrdB / 10)  # noise variance (power)

            for cntr in range(N_c):
                message_bits = np.random.randint(0, 2, int(N * k))  # message
                modulated = modem.modulate(message_bits)  # modulation

                Es = np.mean(np.abs(modulated) ** 2)  # symbol energy
                No = Es / ((10 ** (EbNo / 10)) * np.log2(M))  # noise spectrum density

                noisy = modulated + np.sqrt(No / 2) * \
                        (np.random.randn(modulated.shape[0]) +
                        1j * np.random.randn(modulated.shape[0]))  # AWGN

                demodulated = modem.demodulate(noisy, noise_var=noiseVar)
                NumErr, BER[cntr] = BER_calc(message_bits,
                                            demodulated)  # bit-error ratio
            mean_BER[idxEbNo, idxM] = np.mean(BER, axis=0)  # averaged bit-error ratio

    ''' Theoretical results '''

    BER_theor = np.empty((len(EbNos), len(Ms)))
    for idxM, M in enumerate(Ms):
        BER_theor[:, idxM] = BER_qam(M, EbNos)

    ''' Curves '''

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    plt.semilogy(EbNos, BER_theor[:, 0], 'g-', label='4-QAM (theory)')
    plt.semilogy(EbNos, BER_theor[:, 1], 'b-', label='16-QAM (theory)')
    plt.semilogy(EbNos, BER_theor[:, 2], 'k-', label='64-QAM (theory)')
    plt.semilogy(EbNos, BER_theor[:, 3], 'r-', label='256-QAM (theory)')

    plt.semilogy(EbNos, mean_BER[:, 0], 'g-o', label='4-QAM (simulation)')
    plt.semilogy(EbNos, mean_BER[:, 1], 'b-o', label='16-QAM (simulation)')
    plt.semilogy(EbNos, mean_BER[:, 2], 'k-o', label='64-QAM (simulation)')
    plt.semilogy(EbNos, mean_BER[:, 3], 'r-o', label='256-QAM (simulation)')

    ax.set_ylim(1e-7, 2)
    ax.set_xlim(0, 25.1)

    plt.title("M-QAM")
    plt.xlabel('EbNo (dB)')
    plt.ylabel('BER')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('qam_ber.png')
    return render(request,  'AppDigitalCommunication/modulation.html')
