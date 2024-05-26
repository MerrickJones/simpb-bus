from flask import Flask, request, render_template, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-program', methods=['POST'])
def run_program():
    # Get input values from the form
    input_data = request.form.to_dict()
    
    # Write input data to 'input.txt'
    with open('input.txt', 'w') as f:
        for key, value in input_data.items():
            f.write(f"{value}\n")
    
    # Run the .exe program
    subprocess.run(['./simplified_B.exe'], check=True)
    
    # Read output files
    sett_res = []
    epp_res = []
    
    with open('sett.res', 'r') as f:
        sett_res = f.readlines()
    
    with open('epp.res', 'r') as f:
        epp_res = f.readlines()
    
    # Strip newlines and convert to float
    sett_res = [float(line.strip()) for line in sett_res]
    epp_res = [float(line.strip()) for line in epp_res]
    
    # Send the output data back to the client
    return jsonify({'sett_res': sett_res, 'epp_res': epp_res})

if __name__ == '__main__':
    app.run(debug=True)
