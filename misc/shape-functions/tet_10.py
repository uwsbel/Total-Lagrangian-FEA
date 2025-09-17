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

def save_node_locations_to_latex(filename, node_coords):
    """Converts node coordinates to LaTeX format and saves to a file."""
    latex_lines = []
    for i, coords in enumerate(node_coords, 1):
        u_val = coords['u']
        v_val = coords['v'] 
        w_val = coords['w']
        latex_lines.append(f"\\text{{Node {i}}}: & \\quad ({sympy.latex(u_val)}, {sympy.latex(v_val)}, {sympy.latex(w_val)}) \\\\")
    
    latex_content = "\\begin{align*}\n" + "\n".join(latex_lines) + "\n\\end{align*}"
    
    with open(filename, 'w') as f:
        f.write(latex_content)
    print(f"Saved {filename}")

def generate_tet10_shape_functions():
    """
    Generates the shape functions for a 10-node tetrahedron element using
    the specified basis functions and node coordinates.
    Each node has only position unknowns (no derivatives).
    """
    # 1. Define symbolic variables for the natural coordinates
    u, v, w = sympy.symbols('u v w')

    # 2. Define the basis functions as specified: 1, u, v, w, uv, uw, vw, u², v², w²
    basis_functions = sympy.Matrix([
        1,           # constant
        u,           # linear in u
        v,           # linear in v  
        w,           # linear in w
        u*v,         # bilinear uv
        u*w,         # bilinear uw
        v*w,         # bilinear vw
        u**2,        # quadratic u²
        v**2,        # quadratic v²
        w**2         # quadratic w²
    ])
    num_unknowns = len(basis_functions)
    
    # 3. Define the 10 nodal locations as specified
    node_coords = [
        {'u': 0, 'v': 0, 'w': 0},         # Node 1: (0,0,0)
        {'u': 1, 'v': 0, 'w': 0},         # Node 2: (1,0,0)
        {'u': 0, 'v': 1, 'w': 0},         # Node 3: (0,1,0)
        {'u': 0, 'v': 0, 'w': 1},         # Node 4: (0,0,1)
        {'u': sympy.Rational(1,2), 'v': 0, 'w': 0},  # Node 5: (0.5,0,0)
        {'u': 0, 'v': sympy.Rational(1,2), 'w': 0},  # Node 6: (0,0.5,0)
        {'u': 0, 'v': 0, 'w': sympy.Rational(1,2)},  # Node 7: (0,0,0.5)
        {'u': sympy.Rational(1,2), 'v': sympy.Rational(1,2), 'w': 0}, # Node 8: (0.5,0.5,0)
        {'u': 0, 'v': sympy.Rational(1,2), 'w': sympy.Rational(1,2)}, # Node 9: (0,0.5,0.5)
        {'u': sympy.Rational(1,2), 'v': 0, 'w': sympy.Rational(1,2)}  # Node 10: (0.5,0,0.5)
    ]
    num_nodes = len(node_coords)

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of basis functions: {num_unknowns}")
    print(f"Node coordinates:")
    for i, coords in enumerate(node_coords, 1):
        print(f"  Node {i}: u={coords['u']}, v={coords['v']}, w={coords['w']}")

    # 4. Establish the system of equations by evaluating the basis functions at each node
    # Since we only have position unknowns (not derivatives), we have one condition per node
    B_matrix_T = sympy.zeros(num_nodes, num_unknowns)
    
    # Populate the B_matrix_T (transpose of the B matrix)
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
    print("Inverting the 10x10 B matrix... this may take a moment.")
    B_matrix_inv = B_matrix.inv()
    print("Inversion complete.")
    
    shape_functions = B_matrix_inv * basis_functions

    return shape_functions, basis_functions, B_matrix_inv, B_matrix, node_coords

def verify_tet10_shape_functions(shape_functions, node_coords):
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
    element_type = "tet"
    num_nodes = 10
    
    # --- Setup Output Directory ---
    folder_name = f"{element_type}_{num_nodes}"
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")
    
    # --- Generation ---
    print("--- Generating Shape Functions for 10-Node Tetrahedron Element ---")
    shape_funcs, basis_funcs, B_inv_matrix, B_matrix, node_coords = generate_tet10_shape_functions()

    # --- Verification ---
    print("\n--- Verifying Shape Functions ---")
    verification_mat = verify_tet10_shape_functions(shape_funcs, node_coords)
    is_identity = (verification_mat == sympy.eye(10))
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
        save_node_locations_to_latex(os.path.join(output_dir, 'node_locations.tex'), node_coords)
        files_saved.append('node_locations.tex')
    except Exception as e:
        print(f"Error saving node locations: {e}")
    
    # Save verification status
    verification_status_file = os.path.join(output_dir, 'verification_status.txt')
    try:
        with open(verification_status_file, 'w') as f:
            f.write(f"Element Type: 10-Node Tetrahedron\n")
            f.write(f"Basis functions: 1, u, v, w, uv, uw, vw, u², v², w²\n")
            f.write(f"Number of nodes: {num_nodes}\n")
            f.write(f"Number of unknowns per node: 1 (position only)\n\n")
            f.write(f"Verification matrix is the identity matrix: {is_identity}\n")
            f.write(f"Files successfully generated: {', '.join(files_saved)}\n")
        print(f"Saved {verification_status_file}")
        files_saved.append('verification_status.txt')
    except Exception as e:
        print(f"Error saving verification status: {e}")
    
    print(f"\nPipeline execution complete. Output saved to '{folder_name}/' folder.")
    print(f"Successfully generated {len(files_saved)} files: {', '.join(files_saved)}")
