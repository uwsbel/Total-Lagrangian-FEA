import sympy
import os

def save_to_latex_file(filename, sympy_object):
    """Converts a SymPy object to a LaTeX string and saves it to a file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to LaTeX string using 'plain' mode for compatibility.
    latex_string = sympy.latex(sympy_object, mode='plain')
    
    with open(filename, 'w') as f:
        f.write(latex_string)
    print(f"Saved {filename}")

def save_node_locations_to_latex(filename, node_coords):
    """Converts node coordinates to LaTeX format and saves to a file."""
    u, v, w, L, W = sympy.symbols('u v w L W')
    
    latex_lines = []
    for i, coords in enumerate(node_coords, 1):
        u_val = coords[u]
        v_val = coords[v] 
        w_val = coords[w]
        latex_lines.append(f"\\text{{Node {i}}}: & \\quad ({sympy.latex(u_val)}, {sympy.latex(v_val)}, {sympy.latex(w_val)}) \\\\")
    
    latex_content = "\\begin{align*}\n" + "\n".join(latex_lines) + "\n\\end{align*}"
    
    with open(filename, 'w') as f:
        f.write(latex_content)
    print(f"Saved {filename}")

def generate_shell_shape_functions():
    """
    Generates the shape functions for a 4-node shell element with 4 unknowns per node
    (r, r_u, r_v, r_w), using a monomial basis approach. This corresponds to the
    S3-44 element type.
    """
    # 1. Define symbolic variables for the natural coordinates and element dimensions
    u, v, w = sympy.symbols('u v w')
    L, W = sympy.symbols('L W')

    # 2. Define the monomial basis for a 16-unknown interpolation function.
    # This basis is chosen to handle 4 nodes with 4 unknowns each (r, r_u, r_v, r_w).
    # It extends the ideas from the beam element to a 2D surface.
    basis_functions = sympy.Matrix([
        1, u, v, w,
        u*v, u*w, v*w, u*v*w,
        u**2, v**2, u**2*v, v**2*u,
        u**3, v**3, u**3*v, v**3*u
    ])
    num_unknowns = len(basis_functions)
    
    # 3. Define the nodal locations in the reference (u,v) plane at w=0
    node_coords = [
        {u: -L/2, v: -W/2, w: 0},  # Node 1
        {u:  L/2, v: -W/2, w: 0},  # Node 2
        {u:  L/2, v:  W/2, w: 0},  # Node 3
        {u: -L/2, v:  W/2, w: 0}   # Node 4
    ]
    num_nodes = len(node_coords)

    # 4. Establish the system of equations by evaluating the interpolation function
    #    and its derivatives at the four nodes.
    B_matrix_T = sympy.zeros(num_unknowns, num_unknowns)
    
    # Pre-calculate derivatives of the basis functions
    basis_diff_u = sympy.diff(basis_functions, u)
    basis_diff_v = sympy.diff(basis_functions, v)
    basis_diff_w = sympy.diff(basis_functions, w)

    # Populate the B_matrix_T (transpose of the B matrix)
    for i in range(num_nodes):
        coords = node_coords[i]
        # There are 4 unknowns per node, so we fill 4 rows for each node.
        row_start_index = i * 4
        
        # Condition for the function value 'r'
        B_matrix_T[row_start_index, :] = basis_functions.subs(coords).T
        # Condition for the derivative 'r_u'
        B_matrix_T[row_start_index + 1, :] = basis_diff_u.subs(coords).T
        # Condition for the derivative 'r_v'
        B_matrix_T[row_start_index + 2, :] = basis_diff_v.subs(coords).T
        # Condition for the derivative 'r_w'
        B_matrix_T[row_start_index + 3, :] = basis_diff_w.subs(coords).T
        
    # The B matrix is the transpose of what we just built
    B_matrix = B_matrix_T.T
    
    # 5. Solve for the shape functions by inverting the B matrix.
    #    s(u,v,w) = B^(-1) * b(u,v,w)
    #    We use LU decomposition for inversion as it can be more stable.
    print("Inverting the 16x16 B matrix... this may take a moment.")
    B_matrix_inv = B_matrix.inv()
    print("Inversion complete.")
    
    shape_functions = B_matrix_inv * basis_functions

    return shape_functions, basis_functions, B_matrix_inv, B_matrix, node_coords

def verify_shape_functions(shape_functions):
    """
    Verifies that the generated shape functions satisfy the Kronecker delta property
    at the nodal points and for the nodal derivatives.
    """
    # 1. Define symbolic variables
    u, v, w, L, W = sympy.symbols('u v w L W')
    
    # 2. Define nodal locations
    node_coords = [
        {u: -L/2, v: -W/2, w: 0},  # Node 1
        {u:  L/2, v: -W/2, w: 0},  # Node 2
        {u:  L/2, v:  W/2, w: 0},  # Node 3
        {u: -L/2, v:  W/2, w: 0}   # Node 4
    ]
    num_nodes = len(node_coords)
    num_shape_functions = len(shape_functions)
    
    # 3. Create the verification matrix
    verification_matrix = sympy.zeros(num_shape_functions, num_shape_functions)

    print("Verifying shape functions...")
    for j in range(num_shape_functions): # Iterate through columns (each shape function)
        s_j = shape_functions[j]
        s_j_du = sympy.diff(s_j, u)
        s_j_dv = sympy.diff(s_j, v)
        s_j_dw = sympy.diff(s_j, w)

        for i in range(num_nodes): # Iterate through nodes
            coords = node_coords[i]
            row_start_index = i * 4
            
            # Check conditions for node i against shape function j
            verification_matrix[row_start_index,     j] = sympy.simplify(s_j.subs(coords))
            verification_matrix[row_start_index + 1, j] = sympy.simplify(s_j_du.subs(coords))
            verification_matrix[row_start_index + 2, j] = sympy.simplify(s_j_dv.subs(coords))
            verification_matrix[row_start_index + 3, j] = sympy.simplify(s_j_dw.subs(coords))

    print("Verification complete.")
    return verification_matrix

if __name__ == '__main__':
    # --- Define Element Parameters ---
    element_type = "shell"
    num_nodes = 4
    num_unknowns_per_node = 4
    
    # --- Setup Output Directory ---
    folder_name = f"{element_type}_{num_nodes}{num_unknowns_per_node}"
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")
    
    # --- Generation ---
    print("--- Generating Shape Functions for Shell S3-44 Element ---")
    shape_funcs, basis_funcs, B_inv_matrix, B_matrix, node_coords = generate_shell_shape_functions()

    # --- Verification ---
    print("\n--- Verifying Shape Functions ---")
    verification_mat = verify_shape_functions(shape_funcs)
    is_identity = (verification_mat == sympy.eye(16))
    print(f"Verification successful: {is_identity}")

    # --- Saving Output for LaTeX ---
    print("\n--- Saving Generated LaTeX Code to Files ---")
    save_to_latex_file(os.path.join(output_dir, 'basis_functions.tex'), basis_funcs)
    save_to_latex_file(os.path.join(output_dir, 'B_matrix.tex'), B_matrix)
    save_to_latex_file(os.path.join(output_dir, 'B_inv_matrix.tex'), B_inv_matrix)
    save_to_latex_file(os.path.join(output_dir, 'shape_functions.tex'), shape_funcs)
    save_to_latex_file(os.path.join(output_dir, 'verification_matrix.tex'), verification_mat)
    save_node_locations_to_latex(os.path.join(output_dir, 'node_locations.tex'), node_coords)
    
    verification_status_file = os.path.join(output_dir, 'verification_status.txt')
    with open(verification_status_file, 'w') as f:
        f.write(f"Verification matrix is the identity matrix: {is_identity}\n")
    print(f"Saved {verification_status_file}")
    
    print(f"\nPipeline execution complete. Output saved to '{folder_name}/' folder.")
    print(f"You can now use the .tex files in your LaTeX report.")
