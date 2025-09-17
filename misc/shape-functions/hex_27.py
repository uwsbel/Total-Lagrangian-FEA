import sympy
import os

def save_to_latex_file(filename, sympy_object):
    """Converts a SymPy object to a LaTeX string and saves it to a file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert to LaTeX string using 'plain' mode for compatibility.
        latex_string = sympy.latex(sympy_object, mode='plain')
        
        with open(filename, 'w') as f:
            f.write(latex_string)
        print(f"Saved {filename}")
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def generate_q27_shape_functions():
    """
    Generates shape functions for the Q27 Lagrange hexahedral element.
    - These functions are constructed directly as tensor products of 1D Lagrange polynomials.
    - This method avoids the need for a large matrix inversion.
    - The element operates in a parent coordinate system (xi, eta, zeta) from -1 to 1.
    """
    # 1. Define symbolic variables for the parent coordinate system
    xi, eta, zeta, s = sympy.symbols('xi eta zeta s')

    # 2. Define the 1D quadratic Lagrange polynomials L_i(s) for s in [-1, 1]
    # Note: The provided text uses indices {0, 1, 2} and nodes {-1, 0, 1}.
    L = [
        sympy.Rational(1, 2) * s * (s - 1),  # L_0(s), for node at s = -1
        1 - s**2,                          # L_1(s), for node at s =  0
        sympy.Rational(1, 2) * s * (s + 1)   # L_2(s), for node at s = +1
    ]

    # 3. Generate the 27 shape functions and their corresponding node locations
    # The shape functions N_ijk are the tensor products: L_i(xi) * L_j(eta) * L_k(zeta)
    # The node numbering a = 1 + i + 3j + 9k gives a lexicographic order.
    shape_functions = []
    node_coords = []
    
    print("Generating 27 Lagrange shape functions...")
    for k in range(3):        # Corresponds to zeta coordinate {-1, 0, 1}
        for j in range(3):    # Corresponds to eta coordinate {-1, 0, 1}
            for i in range(3):# Corresponds to xi coordinate {-1, 0, 1}
                
                # Create the shape function N_ijk
                N_ijk = L[i].subs(s, xi) * L[j].subs(s, eta) * L[k].subs(s, zeta)
                shape_functions.append(sympy.expand(N_ijk))
                
                # Define the coordinate for the node (i, j, k)
                # The node indices {0, 1, 2} map to coordinates {-1, 0, 1}
                coord = (i - 1, j - 1, k - 1)
                node_coords.append(coord)
    
    print("Shape function generation complete.")
    return sympy.Matrix(shape_functions), node_coords

def verify_q27_shape_functions(shape_functions, node_coords):
    """
    Verifies the Kronecker-delta property of the Q27 shape functions.
    N_j(at node_i) should be 1 if i=j and 0 if i!=j.
    """
    xi, eta, zeta = sympy.symbols('xi eta zeta')
    num_nodes = len(shape_functions)
    verification_matrix = sympy.zeros(num_nodes, num_nodes)

    print("Verifying shape functions...")
    try:
        # Iterate through each shape function N_j (columns of the matrix)
        for j in range(num_nodes):
            s_j = shape_functions[j]
            
            # Iterate through each node_i (rows of the matrix)
            for i in range(num_nodes):
                # Get the coordinates of node_i
                coords_i = {
                    xi: node_coords[i][0],
                    eta: node_coords[i][1],
                    zeta: node_coords[i][2]
                }
                
                # Substitute the coordinates into the shape function and simplify
                verification_matrix[i, j] = sympy.simplify(s_j.subs(coords_i))

        print("Verification complete.")
        return verification_matrix, True
    except Exception as e:
        print(f"Error during verification: {e}")
        return sympy.Matrix([]), False


if __name__ == '__main__':
    # --- Define Element Parameters ---
    element_type = "hex"
    folder_name = "hex_q27_lagrange"
    
    # --- Setup Output Directory ---
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")
    
    # --- Generation ---
    print(f"--- Generating Shape Functions for {folder_name.upper()} Element ---")
    shape_funcs, node_locs = generate_q27_shape_functions()

    # --- Verification ---
    print(f"\n--- Verifying Shape Functions ---")
    verification_mat, verification_successful = verify_q27_shape_functions(shape_funcs, node_locs)
    
    is_identity = False
    if verification_successful:
        is_identity = (verification_mat == sympy.eye(27))
        print(f"Verification successful: The result is the identity matrix -> {is_identity}")
    else:
        print("Verification failed or could not be completed.")

    # --- Saving Output for LaTeX ---
    print("\n--- Saving Generated LaTeX Code to Files ---")
    files_saved = []
    
    # Save 1D Lagrange Polynomials
    try:
        s = sympy.symbols('s')
        L_1D = sympy.Matrix([
            sympy.Rational(1, 2) * s * (s - 1),
            1 - s**2,
            sympy.Rational(1, 2) * s * (s + 1)
        ])
        if save_to_latex_file(os.path.join(output_dir, '1D_lagrange_polynomials.tex'), L_1D):
            files_saved.append('1D_lagrange_polynomials.tex')
    except Exception as e:
        print(f"Error saving 1D Lagrange polynomials: {e}")

    # Save shape functions
    if save_to_latex_file(os.path.join(output_dir, 'shape_functions.tex'), shape_funcs):
        files_saved.append('shape_functions.tex')
    
    # Save verification matrix
    if save_to_latex_file(os.path.join(output_dir, 'verification_matrix.tex'), verification_mat):
        files_saved.append('verification_matrix.tex')
    
    # Save node locations
    try:
        node_locations_file = os.path.join(output_dir, 'node_locations.tex')
        with open(node_locations_file, 'w') as f:
            f.write("\\begin{verbatim}\n")
            f.write("Node # | (i,j,k) | (xi, eta, zeta)\n")
            f.write("----------------------------------\n")
            node_num = 1
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        f.write(f"{node_num:<6} | ({i},{j},{k})     | ({i-1: >2}, {j-1: >2}, {k-1: >2})\n")
                        node_num += 1
            f.write("\\end{verbatim}")
        print(f"Saved {node_locations_file}")
        files_saved.append('node_locations.tex')
    except Exception as e:
        print(f"Error saving node locations: {e}")
    
    # Save verification status
    verification_status_file = os.path.join(output_dir, 'verification_status.txt')
    try:
        with open(verification_status_file, 'w') as f:
            f.write(f"Element Type: 27-Node Lagrange Hexahedron (Q27)\n")
            f.write(f"Basis: Tensor product of 1D Quadratic Lagrange Polynomials\n\n")
            f.write(f"Verification matrix is the identity matrix: {is_identity}\n")
            f.write(f"Verification process successful: {verification_successful}\n")
            f.write(f"Files successfully generated: {', '.join(files_saved)}\n")
        print(f"Saved {verification_status_file}")
        files_saved.append('verification_status.txt')
    except Exception as e:
        print(f"Error saving verification status: {e}")
    
    print(f"\nPipeline execution complete. Output saved to '{folder_name}/' folder.")
    print(f"Successfully generated {len(files_saved)} files: {', '.join(files_saved)}")