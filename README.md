# MICROPYTHON LSTM EXAMPLE

This example demonstrates how to estimate state of health and state of chrage of a Li-ion battery using LSTM and micropython on esp32.


## Prerequisites

- Install [micropython and microlite](https://github.com/adithyab94/tensorflow-micropython-examples)
- Erase flash on esp32 
- Install the firmware on esp32 using write_flash
- copy the relevant files
- run

## Generate test data

- Run the following command to generate test data
```python
input_cols = ['U[V]', 'I[A]', 'T_amb'] 
output_cols = 'SoH'

data = pd.read_csv("data_fraun.csv", usecols=input_cols)
trimmed_df = data.iloc[:300]

# Convert the trimmed DataFrame to a NumPy array
data_array = trimmed_df.values

# Reshape the array to (16, 64, 5)
reshaped_array = data_array.reshape(1, 300, 3)
np.savetxt('input.dat', reshaped_array.astype(np.float32).flatten(), fmt='%f')
```

## Run the example

- Run the following command to run the example
```bash
esptool.py --chip esp32s3 --port COM8 erase_flash
esptool.py --chip esp32s3 --port COM8 write_flash -z 0 firmware-esp32s3-cam.bin
```

- use ampy package or download pyboard from micorpython repo
https://github.com/micropython/micropython/blob/master/tools/pyboard.py

- Copy files
```bash
python pyboard.py --device COM8 -f cp input.dat :
python pyboard.py --device COM8 -f cp keras_soc.tflite :
python pyboard.py --device COM8 -f cp soh_model.tflite :
python pyboard.py --device COM8 -f cp main.py :
```

- Run

```bash
python pyboard.py --device COM8 main.py
```

