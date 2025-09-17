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

def generate_hex27_monomial_shape_functions():
    """
    Generates shape functions for the 27-node hexahedral element using
    explicit monomial basis functions instead of Lagrange polynomials.
    Each node has only position unknowns (no derivatives).
    """
    # 1. Define symbolic variables for the natural coordinates and element dimensions
    u, v, w = sympy.symbols('u v w')
    L, W, H = sympy.symbols('L W H')

    # 2. Define the explicit monomial basis as specified
    explicit_basis = [
        # i=0, j=0, k=0,1,2
        1, w, w**2,
        
        # i=0, j=1, k=0,1,2
        v, v*w, v*w**2,
        
        # i=0, j=2, k=0,1,2
        v**2, v**2*w, v**2*w**2,
        
        # i=1, j=0, k=0,1,2
        u, u*w, u*w**2,
        
        # i=1, j=1, k=0,1,2
        u*v, u*v*w, u*v*w**2,
        
        # i=1, j=2, k=0,1,2
        u*v**2, u*v**2*w, u*v**2*w**2,
        
        # i=2, j=0, k=0,1,2
        u**2, u**2*w, u**2*w**2,
        
        # i=2, j=1, k=0,1,2
        u**2*v, u**2*v*w, u**2*v*w**2,
        
        # i=2, j=2, k=0,1,2
        u**2*v**2, u**2*v**2*w, u**2*v**2*w**2
    ]
    
    basis_functions = sympy.Matrix(explicit_basis)
    num_unknowns = len(basis_functions)
    
    print(f"Number of basis functions: {num_unknowns}")

    # 3. Define the 27 nodal locations in the same order as hex_27.py
    # Node coordinates in parent coordinate system (u, v, w) using parametric dimensions L, W, H
    node_coords = []
    
    print("Generating 27 node coordinates...")
    # Define the coordinate values for each dimension
    u_coords = [-L/2, 0, L/2]   # Corresponds to u coordinate {-L/2, 0, L/2}
    v_coords = [-W/2, 0, W/2]   # Corresponds to v coordinate {-W/2, 0, W/2}
    w_coords = [-H/2, 0, H/2]   # Corresponds to w coordinate {-H/2, 0, H/2}
    
    for k in range(3):        # Corresponds to w coordinate
        for j in range(3):    # Corresponds to v coordinate
            for i in range(3):# Corresponds to u coordinate
                coord = {'u': u_coords[i], 'v': v_coords[j], 'w': w_coords[k]}
                node_coords.append(coord)
    
    num_nodes = len(node_coords)
    print(f"Number of nodes: {num_nodes}")
    
    print(f"Node coordinates:")
    for i, coords in enumerate(node_coords, 1):
        print(f"  Node {i}: u={coords['u']}, v={coords['v']}, w={coords['w']}")

    # 4. Establish the system of equations by evaluating the basis functions at each node
    # Since we only have position unknowns (not derivatives), we have one condition per node
    B_matrix_T = sympy.zeros(num_nodes, num_unknowns)
    
    # Populate the B_matrix_T (transpose of the B matrix)
    print("Building B matrix...")
    for i in range(num_nodes):
        coords = node_coords[i]
        substitution_dict = {u: coords['u'], v: coords['v'], w: coords['w']}
        # Condition for the function value at node i
        B_matrix_T[i, :] = basis_functions.subs(substitution_dict).T
        
    # The B matrix is the transpose of what we just built
    B_matrix = B_matrix_T.T
    
    print(f"B matrix shape: {B_matrix.shape}")
    
    # 5. Solve for the shape functions by inverting the B matrix.
    #    N(u,v,w) = B^(-1) * basis(u,v,w)
    print("Inverting the 27x27 B matrix... this may take a moment.")
    B_matrix_inv = B_matrix.inv()
    print("Inversion complete.")
    
    shape_functions = B_matrix_inv * basis_functions

    return shape_functions, basis_functions, B_matrix_inv, B_matrix, node_coords

def verify_hex27_shape_functions(shape_functions, node_coords):
    """
    Verifies that the generated shape functions satisfy the Kronecker delta property:
    N_j(node_i) should be 1 if i=j and 0 if i!=j.
    """
    # 1. Define symbolic variables
    u, v, w = sympy.symbols('u v w')
    
    num_nodes = len(node_coords)
    num_shape_functions = len(shape_functions)
    
    # 2. Create the verification matrix
    verification_matrix = sympy.zeros(num_nodes, num_shape_functions)

    print("Verifying shape functions...")
    for j in range(num_shape_functions): # Iterate through columns (each shape function)
        s_j = shape_functions[j]

        for i in range(num_nodes): # Iterate through nodes
            coords = node_coords[i]
            substitution_dict = {u: coords['u'], v: coords['v'], w: coords['w']}
            
            # Check condition: N_j evaluated at node i should be δ_ij
            verification_matrix[i, j] = sympy.simplify(s_j.subs(substitution_dict))

    print("Verification complete.")
    return verification_matrix

if __name__ == '__main__':
    # --- Define Element Parameters ---
    element_type = "hex"
    folder_name = "hex_27_monomial"
    
    # --- Setup Output Directory ---
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")
    
    # --- Generation ---
    print("--- Generating Shape Functions for 27-Node Hexahedron with Monomial Basis ---")
    shape_funcs, basis_funcs, B_inv_matrix, B_matrix, node_coords = generate_hex27_monomial_shape_functions()

    # --- Verification ---
    print("\n--- Verifying Shape Functions ---")
    verification_mat = verify_hex27_shape_functions(shape_funcs, node_coords)
    is_identity = (verification_mat == sympy.eye(27))
    print(f"Verification successful: The result is the identity matrix -> {is_identity}")

    # --- Saving Output for LaTeX ---
    print("\n--- Saving Generated LaTeX Code to Files ---")
    files_saved = []
    
    # Save basis functions
    if save_to_latex_file(os.path.join(output_dir, 'basis_functions.tex'), basis_funcs):
        files_saved.append('basis_functions.tex')
    
    # Save B matrix
    if save_to_latex_file(os.path.join(output_dir, 'B_matrix.tex'), B_matrix):
        files_saved.append('B_matrix.tex')
    
    # Save B inverse matrix
    if save_to_latex_file(os.path.join(output_dir, 'B_inv_matrix.tex'), B_inv_matrix):
        files_saved.append('B_inv_matrix.tex')
    
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
            f.write("Node # | (i,j,k) | (u, v, w) coordinates\n")
            f.write("--------------------------------------------\n")
            node_num = 1
            coord_labels = ['-L/2', '0', 'L/2']  # For u coordinates
            coord_labels_v = ['-W/2', '0', 'W/2']  # For v coordinates  
            coord_labels_w = ['-H/2', '0', 'H/2']  # For w coordinates
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        f.write(f"{node_num:<6} | ({i},{j},{k})     | ({coord_labels[i]}, {coord_labels_v[j]}, {coord_labels_w[k]})\n")
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
            f.write(f"Element Type: 27-Node Hexahedron with Monomial Basis\n")
            f.write(f"Basis: Explicit monomial functions (not Lagrange polynomials)\n")
            f.write(f"Number of nodes: 27\n")
            f.write(f"Number of unknowns per node: 1 (position only)\n")
            f.write(f"Coordinates: Parametric with dimensions L, W, H\n")
            f.write(f"Node layout: 3x3x3 grid with coordinates {{-L/2, 0, L/2}} x {{-W/2, 0, W/2}} x {{-H/2, 0, H/2}}\n\n")
            f.write(f"Verification matrix is the identity matrix: {is_identity}\n")
            f.write(f"Files successfully generated: {', '.join(files_saved)}\n")
        print(f"Saved {verification_status_file}")
        files_saved.append('verification_status.txt')
    except Exception as e:
        print(f"Error saving verification status: {e}")
    
    print(f"\nPipeline execution complete. Output saved to '{folder_name}/' folder.")
    print(f"Successfully generated {len(files_saved)} files: {', '.join(files_saved)}")
