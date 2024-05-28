from flask import Flask, request, render_template, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-program', methods=['POST'])
def run_program():
    input_data = request.form.to_dict()

    layers = int(input_data.get('layers', 0))
    stages = int(input_data.get('stages', 0))
    depth_points = int(input_data.get('depthPoints', 0))
    sett_out = int(input_data.get('settOut', 0))
    epp_zero = int(input_data.get('eppZero', 0))

    input_content = f"{input_data['layers']} {input_data['alpha']} {input_data['beta']}\n"

    # Table 1
    for i in range(layers):
        row_data = []
        for j in range(7):
            row_data.append(input_data.get(f'table1_row{i}_col{j}', ''))
        input_content += ' '.join(row_data).strip() + '\n'

    # Table 2
    for i in range(layers):
        row_data = []
        for j in range(5):
            row_data.append(input_data.get(f'table2_row{i}_col{j}', ''))
        input_content += ' '.join(row_data).strip() + '\n'

    input_content += f"{input_data['stages']}\n"

    # Table 3
    for i in range(stages):
        row_data = []
        for j in range(5):
            row_data.append(input_data.get(f'table3_row{i}_col{j}', ''))
        input_content += ' '.join(row_data).strip() + '\n'

    input_content += f"{input_data['terms']} {input_data['drainage']} {input_data['unitWeight']}\n"

    # Table 4
    for i in range(layers):
        row_data = []
        for j in range(7):
            row_data.append(input_data.get(f'table4_row{i}_col{j}', ''))
        input_content += ' '.join(row_data).strip() + '\n'

    input_content += f"{input_data['depthPoints']}\n"

    # Table 5
    for i in range(depth_points):
        input_content += input_data.get(f'table5_row{i}_col0', '').strip() + ' '

    input_content += '\n'
    input_content += f"{input_data['settOut']}\n"

    # Table 6
    for i in range(sett_out):
        input_content += input_data.get(f'table6_row{i}_col0', '').strip() + ' '

    input_content += '\n'
    input_content += f"{input_data['eppZero']}\n"

    # Table 7
    for i in range(2 + epp_zero * 2):
        input_content += input_data.get(f'table7_row{i}_col0', '').strip() + ' '

    input_content += '\n'

    with open('input.txt', 'w') as f:
        f.write(input_content)

    try:
        subprocess.run(['simplified_B.exe'], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

    try:
        with open('sett.res', 'r') as f:
            sett_res = f.readlines()
        with open('epp.res', 'r') as f:
            epp_res = f.readlines()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

    # Preparing the data to display in the required format
    sett_res_data = [line.strip().split() for line in sett_res if line.strip()]
    epp_res_data = [line.strip().split() for line in epp_res if line.strip()]

    return render_template('output.html', sett_res_data=sett_res_data, epp_res_data=epp_res_data)

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)
        #app.run(debug=True)
    
