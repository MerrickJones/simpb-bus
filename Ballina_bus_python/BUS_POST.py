import os
import shutil
import subprocess
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

def create_folder_and_copy_files(base_folder, iteration, input_files, exe_file_path):
    new_folder_path = os.path.join(base_folder, f"iteration_{iteration}")
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Created folder: {new_folder_path}")

    for input_file in input_files:
        shutil.copy(input_file, os.path.join(new_folder_path, os.path.basename(input_file)))
        print(f"Copied {input_file} to {new_folder_path}")

    shutil.copy(exe_file_path, new_folder_path)
    print(f"Copied {exe_file_path} to {new_folder_path}")
    return new_folder_path

def call_executable(i, base_folder, input_files, exe_file_path):
    new_folder_path = create_folder_and_copy_files(base_folder, i, input_files, exe_file_path)
    exe_path = os.path.join(new_folder_path, os.path.basename(exe_file_path))

    try:
        subprocess.run([exe_path], cwd=new_folder_path, check=True)
        output_file_path = os.path.join(new_folder_path, "XX10.res")
        matrix = parse_output_to_matrix(output_file_path)
        return (i, matrix)
    except subprocess.CalledProcessError as e:
        print(f"Error running {exe_path} in iteration {i}: {e.stderr}")
        return (i, None)

def parse_output_to_matrix(output_file_path):
    matrix = []
    with open(output_file_path, 'r') as f:
        for line in f:
            if line.strip():
                matrix.append(list(map(float, line.split())))
    return np.array(matrix, dtype=np.float32)


def save_all_matrices(results, output_file):
    with open(output_file, 'w') as f:
        for i, matrix in results:
            if matrix is not None:
                np.savetxt(f, matrix, fmt='%.6f', delimiter=" ")
                print(f"Matrix from iteration {i} saved in {output_file}.")


def run_second_executable(i, base_folder, matrix, settlement):
    new_folder_path = os.path.join(base_folder, f"second_iteration_{i}")
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Created second iteration folder: {new_folder_path}")

    dat1 = np.loadtxt('dat1.txt', dtype=np.float32)
    dat3 = np.loadtxt('dat3.txt', dtype=np.float32)
    dat4 = np.loadtxt('dat4.txt', dtype=np.float32)

    n_lay = 6
    nlay = 9
    x = matrix
    dat2 = np.zeros((9, 5), dtype=np.float32)

    for idx in range(4):
        dat2[idx, 0] = x[idx]
        dat2[idx, 1] = dat2[idx, 0] * x[18]
        dat2[idx, 2] = x[idx + n_lay]
        dat2[idx, 3] = dat2[idx, 2]
        dat2[idx, 4] = dat2[idx, 1] * x[19]

    dat2[4] = dat2[3]

    for idx in range(5, 7):
        dat2[idx, 0] = x[idx - 1]
        dat2[idx, 1] = dat2[idx, 0] * x[18]
        dat2[idx, 2] = x[idx + n_lay - 1]
        dat2[idx, 3] = dat2[idx, 2]
        dat2[idx, 4] = dat2[idx, 1] * x[19]

    dat2[7] = dat2[6]
    dat2[8] = dat2[7]

    for idx in range(1, nlay + 1):
        dat1[idx, 4] *= x[20]

    filename2 = os.path.join(new_folder_path, 'dat2.txt')
    np.savetxt(filename2, dat2, fmt='%g', delimiter='\t')

    filename1 = os.path.join(new_folder_path, 'dat11.txt')
    np.savetxt(filename1, dat1, fmt='%g', delimiter='\t')

    shutil.copy('dat3.txt', new_folder_path)
    shutil.copy('dat4.txt', new_folder_path)

    combined_filename = os.path.join(new_folder_path, 'input.txt')
    with open(combined_filename, 'wb') as outfile:
        for fname in [filename1, filename2, os.path.join(new_folder_path, 'dat3.txt'),
                      os.path.join(new_folder_path, 'dat4.txt')]:
            with open(fname, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)

    exe_path = os.path.join(new_folder_path, 'simplified_B.exe')
    shutil.copy('simplified_B.exe', exe_path)

    try:
        os.chmod(exe_path, 0o755)
        subprocess.run([exe_path], cwd=new_folder_path, check=True)
        output_file_path = os.path.join(new_folder_path, "sett.txt")
        sett = parse_output_to_matrix(output_file_path)

        Y_sim = sett[:, 1:5].flatten()
        diff = np.sqrt(np.sum((settlement - Y_sim) ** 2))
        return (i, diff)
    except subprocess.CalledProcessError as e:
        print(f"Error running {exe_path} in iteration {i}: {e.stderr}")
        return (i, None)
    except PermissionError as e:
        print(f"Permission error for {exe_path}: {e}")
        return (i, None)

def find_best_iteration(results):
    min_diff = float('inf')
    best_index = -1
    for i, diff in results:
        if diff is not None and diff < min_diff:
            min_diff = diff
            best_index = i
    return best_index

def cleanup(base_folder):
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
        print(f"Cleaned up and removed the folder: {base_folder}")

def plot_results(best_iteration_folder):
    sett = np.loadtxt(os.path.join(best_iteration_folder, 'sett.txt'), dtype=np.float32)
    Y_sim1 = sett[:, 1]
    Y_sim2 = sett[:, 2]
    Y_sim3 = sett[:, 3]
    Y_sim4 = sett[:, 4]

    sett_0 = np.loadtxt('sett_0.txt', dtype=np.float32)
    sett_1 = np.loadtxt('sett_1.txt', dtype=np.float32)
    sett_2 = np.loadtxt('sett_2.txt', dtype=np.float32)
    sett_3 = np.loadtxt('sett_3.txt', dtype=np.float32)

    plt.figure(1)
    plt.plot(Y_sim1, 'b+', label='Y_sim1')
    plt.plot(np.abs(sett_0), 'r-', label='sett_0')

    plt.plot(Y_sim2, 'bs', label='Y_sim2')
    plt.plot(np.abs(sett_1), 'r:', label='sett_1')

    plt.plot(Y_sim3, 'bo', label='Y_sim3')
    plt.plot(np.abs(sett_2), 'r--', label='sett_2')

    plt.plot(Y_sim4, 'b*', label='Y_sim4')
    plt.plot(np.abs(sett_3), 'r--o', label='sett_3')

    plt.legend()
    # Save the figure
    plt.savefig('sample_plot.png')
    plt.show()

def main():
    base_folder = "results"
    input_files = ["dat1.txt", "dat2.txt", "dat3.txt", "dat4.txt", "bus.txt", "sett.txt", "epp.txt",
                   "MathNet.Numerics.dll"]
    exe_file_path = "BUS.exe"
    num_iterations = 10
    output_file = "all_matrices.txt"

    os.makedirs(base_folder, exist_ok=True)


    # Uncomment this block to run the first set of iterations
    with mp.Pool(processes=num_iterations) as pool:
        args_list = [(i, base_folder, input_files, exe_file_path) for i in range(num_iterations)]
        results = pool.starmap(call_executable, args_list)

    save_all_matrices(results, output_file)

    cleanup(base_folder)
    os.makedirs(base_folder, exist_ok=True)

    all_matrices = np.loadtxt(output_file, dtype=np.float32)
    settlement = np.loadtxt('settlement.txt', dtype=np.float32)

    # Sort all_matrices by the first column in descending order
    all_matrices = all_matrices[np.argsort(all_matrices[:, 0])[::-1]]

    final_results = []
    previous_matrix = None

    with mp.Pool(processes=min(len(all_matrices), 10)) as pool:
        for idx, matrix in enumerate(all_matrices):
            if previous_matrix is not None and np.array_equal(previous_matrix, matrix):
                continue
            previous_matrix = matrix
            result = pool.apply_async(run_second_executable, (idx, base_folder, matrix, settlement))
            final_results.append(result)
        final_results = [result.get() for result in final_results]

    best_index = find_best_iteration(final_results)
    print(f"Best iteration index: {best_index}")

    best_iteration_folder = os.path.join(base_folder, f"second_iteration_{best_index}")
    plot_results(best_iteration_folder)
    cleanup(base_folder)

if __name__ == "__main__":
    main()
