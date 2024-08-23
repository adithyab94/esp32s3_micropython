import microlite

# Set the mode (if required by your application)
mode = 1

# Initialize the test image array
test_soh = bytearray(4 * 8 * 4)
test_soc = bytearray(1 * 300 * 3)

def input_callback_soh(microlite_interpreter):
    inputTensor = microlite_interpreter.getInputTensor(0)

    for i in range(len(test_soh)):
        inputTensor.setValue(i, test_soh[i])
    
    print("Setup %d bytes on the inputTensor." % len(test_soh))

def output_callback_soh(microlite_interpreter):
    outputTensor = microlite_interpreter.getOutputTensor(0)
    output_values = []

    for i in range(4):
        output_values.append(outputTensor.getValue(i))
    
    print("Output values:", output_values)
    
def input_callback_soc(microlite_interpreter):
    inputTensor = microlite_interpreter.getInputTensor(0)

    for i in range(len(test_soc)):
        inputTensor.setValue(i, test_soc[i])
    
    print("Setup %d bytes on the inputTensor." % len(test_soc))

def output_callback_soc(microlite_interpreter):
    outputTensor = microlite_interpreter.getOutputTensor(0)
    output_values = []

    for i in range(4):
        output_values.append(outputTensor.getValue(i))
    
    print("Output values:", output_values)

# Load the model
person_detection_model_file = open('soh-model.tflite', 'rb')
person_detection_model = bytearray(300568)
person_detection_model_file.readinto(person_detection_model)
person_detection_model_file.close()

# Initialize the interpreter
interp_soh = microlite.interpreter(person_detection_model, 136*1024, input_callback_soh, output_callback_soh)


# Load the model
soc_file = open('keras_soc.tflite', 'rb')
soc_model = bytearray(300568)
soc_file.readinto(soc_model)
soc_file.close()

# Initialize the interpreter
interp = microlite.interpreter(soc_model, 300*1024, input_callback_soc, output_callback_soc)

# Function to classify an image
def soh(image_file_path):
    with open(image_file_path, 'rb') as image_file:
        image_file.readinto(test_soh)
    
    interp_soh.invoke()
    
def soc(image_file_path):
    with open(image_file_path, 'rb') as image_file:
        image_file.readinto(test_soc)
    
    interp.invoke()

# Classify custom reshaped image from input.dat
print("SOH from input.dat")
soh('input.dat')

print("SOC from input.dat")
soc('input.dat')